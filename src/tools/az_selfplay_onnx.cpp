#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "action.h"
#include "bitboard.h"
#include "board.h"
#include "game.h"
#include "piece.h"
#include "state.h"
#include "vec4.h"

#include "onnxruntime_cxx_api.h"

namespace {

bool diag_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("AZ_ONNX_DIAG");
        if (value == nullptr) {
            return false;
        }
        return std::string(value) != "0";
    }();
    return enabled;
}

void diag_log(const std::string& message) {
    if (!diag_enabled()) {
        return;
    }
    std::cerr << "[az_selfplay_onnx] " << message << "\n" << std::flush;
}

constexpr int kDefaultPieceChannels = 14;
constexpr int kDefaultLineShift = 5;
constexpr const char* kDefaultVariantPgn = "[Board \"Very Small - Open\"]\n[Mode \"5D\"]\n";

struct Semimove {
    int line_idx = 0;
    vec4 from{0, 0, 0, 0};
    vec4 to{0, 0, 0, 0};

    [[nodiscard]] auto sort_key() const {
        return std::make_tuple(
            line_idx,
            from.x(), from.y(), from.t(), from.l(),
            to.x(), to.y(), to.t(), to.l()
        );
    }

    [[nodiscard]] bool operator==(const Semimove& other) const = default;

    [[nodiscard]] ext_move to_ext_move() const {
        return ext_move(from, to);
    }
};

struct BoardKey {
    int l = 0;
    int t = 0;
    bool c = false;
};

struct EncodedState {
    std::vector<float> board_planes;
    std::vector<float> last_move_markers;
    std::vector<int64_t> l_coords;
    std::vector<int64_t> t_coords;
    std::vector<BoardKey> board_keys;
    int num_boards = 0;
};

struct ActionChoice {
    bool is_submit = false;
    Semimove semimove{};
};

struct ActionEntry {
    ActionChoice action;
    int64_t board_idx = -1;
    int64_t from_sq = 0;
    int64_t to_sq = 0;
    float delta_t = 0.0f;
    float delta_l = 0.0f;
    int64_t is_submit = 0;
};

struct SearchConfig {
    std::string variant_name = "very_small";
    std::string variant_pgn = kDefaultVariantPgn;
    int board_side = 4;
    int board_squares = 16;
    int piece_channels = kDefaultPieceChannels;
    int line_shift = kDefaultLineShift;
    int num_simulations = 200;
    int leaf_batch_size = 1;
    float c_puct = 2.0f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;
    float temperature = 1.0f;
    float temperature_final = 0.1f;
    int temperature_threshold = 30;
    int min_board_limit = 15;
    int max_board_limit = 25;
    float material_scale = 2.0f;
    int max_game_length = 0;
    int num_games = 1;
    std::string provider = "cpu";
    int cuda_device_id = 0;
    int ort_intra_threads = 1;
    bool use_transposition_table = true;
    uint32_t seed = 1;
    bool serve_mode = false;
    bool print_games = true;
    std::filesystem::path output_data_path;
    std::filesystem::path profile_json_path;
};

struct ProfileStat {
    double total_sec = 0.0;
    uint64_t calls = 0;

    void add(double elapsed_sec) {
        total_sec += elapsed_sec;
        calls += 1;
    }
};

struct RunnerProfile {
    ProfileStat run_batch;
    ProfileStat play_game;
    ProfileStat mcts_select_action;
    ProfileStat mcts_simulation;
    ProfileStat tree_select;
    ProfileStat env_clone;
    ProfileStat replay_apply_semimove;
    ProfileStat replay_submit_turn;
    ProfileStat mcts_backprop;
    ProfileStat expand_node;
    ProfileStat tt_lookup;
    ProfileStat legal_frontier;
    ProfileStat encode_state;
    ProfileStat build_action_entries;
    ProfileStat onnx_predict_actions;
    ProfileStat softmax;
    ProfileStat sample_policy;
    ProfileStat selfplay_apply_semimove;
    ProfileStat selfplay_submit_turn;
    ProfileStat format_move;
    ProfileStat show_pgn;
    ProfileStat binary_write_game;

    uint64_t games = 0;
    uint64_t semimoves = 0;
    uint64_t samples = 0;
    uint64_t simulations = 0;
    uint64_t network_calls = 0;
    uint64_t frontier_nodes = 0;
    uint64_t tt_hits = 0;
    uint64_t tt_misses = 0;
};

[[nodiscard]] double seconds_since(std::chrono::steady_clock::time_point start) {
    using seconds = std::chrono::duration<double>;
    return std::chrono::duration_cast<seconds>(std::chrono::steady_clock::now() - start).count();
}

[[nodiscard]] std::string json_escape(const std::string& value) {
    std::ostringstream oss;
    for (char ch : value) {
        switch (ch) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20U) {
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
                } else {
                    oss << ch;
                }
                break;
        }
    }
    return oss.str();
}

void write_profile_json(
    const std::filesystem::path& path,
    const RunnerProfile& profile,
    const SearchConfig& cfg
) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Could not open profile JSON output file: " + path.string());
    }

    const auto write_stat = [&](const char* name, const ProfileStat& stat, bool trailing_comma) {
        out << "    \"" << name << "\": {\"calls\": " << stat.calls
            << ", \"total_sec\": " << std::fixed << std::setprecision(6) << stat.total_sec;
        if (stat.calls > 0) {
            out << ", \"avg_sec\": " << (stat.total_sec / static_cast<double>(stat.calls));
        } else {
            out << ", \"avg_sec\": 0.0";
        }
        out << "}";
        if (trailing_comma) {
            out << ",";
        }
        out << "\n";
    };

    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"variant\": \"" << json_escape(cfg.variant_name) << "\",\n";
    out << "    \"games\": " << cfg.num_games << ",\n";
    out << "    \"sims\": " << cfg.num_simulations << ",\n";
    out << "    \"leaf_batch_size\": " << cfg.leaf_batch_size << ",\n";
    out << "    \"provider\": \"" << json_escape(cfg.provider) << "\",\n";
    out << "    \"ort_threads\": " << cfg.ort_intra_threads << ",\n";
    out << "    \"min_board_limit\": " << cfg.min_board_limit << ",\n";
    out << "    \"max_board_limit\": " << cfg.max_board_limit << ",\n";
    out << "    \"max_game_length\": " << cfg.max_game_length << "\n";
    out << "  },\n";
    out << "  \"totals\": {\n";
    out << "    \"games\": " << profile.games << ",\n";
    out << "    \"semimoves\": " << profile.semimoves << ",\n";
    out << "    \"samples\": " << profile.samples << ",\n";
    out << "    \"simulations\": " << profile.simulations << ",\n";
    out << "    \"network_calls\": " << profile.network_calls << ",\n";
    out << "    \"frontier_nodes\": " << profile.frontier_nodes << ",\n";
    out << "    \"tt_hits\": " << profile.tt_hits << ",\n";
    out << "    \"tt_misses\": " << profile.tt_misses << "\n";
    out << "  },\n";
    out << "  \"timing\": {\n";
    write_stat("run_batch", profile.run_batch, true);
    write_stat("play_game", profile.play_game, true);
    write_stat("mcts_select_action", profile.mcts_select_action, true);
    write_stat("mcts_simulation", profile.mcts_simulation, true);
    write_stat("tree_select", profile.tree_select, true);
    write_stat("env_clone", profile.env_clone, true);
    write_stat("replay_apply_semimove", profile.replay_apply_semimove, true);
    write_stat("replay_submit_turn", profile.replay_submit_turn, true);
    write_stat("mcts_backprop", profile.mcts_backprop, true);
    write_stat("expand_node", profile.expand_node, true);
    write_stat("tt_lookup", profile.tt_lookup, true);
    write_stat("legal_frontier", profile.legal_frontier, true);
    write_stat("encode_state", profile.encode_state, true);
    write_stat("build_action_entries", profile.build_action_entries, true);
    write_stat("onnx_predict_actions", profile.onnx_predict_actions, true);
    write_stat("softmax", profile.softmax, true);
    write_stat("sample_policy", profile.sample_policy, true);
    write_stat("selfplay_apply_semimove", profile.selfplay_apply_semimove, true);
    write_stat("selfplay_submit_turn", profile.selfplay_submit_turn, true);
    write_stat("format_move", profile.format_move, true);
    write_stat("show_pgn", profile.show_pgn, true);
    write_stat("binary_write_game", profile.binary_write_game, false);
    out << "  }\n";
    out << "}\n";
}

struct MoveLogEntry {
    int player = 0;
    bool is_submit = false;
    std::string move_text;
    float root_value = 0.0f;
    int board_count = 0;
};

struct GameResult {
    float outcome = 0.0f;
    int total_semimoves = 0;
    int board_limit = 0;
    std::string terminal_reason;
    std::string pgn;
    std::vector<MoveLogEntry> move_history;
    struct TrainingSample {
        EncodedState encoded_state;
        std::vector<ActionEntry> action_entries;
        std::vector<float> policy_probs;
        float urgency = 0.0f;
        float value_target = 0.0f;
        int player = 0;
    };
    std::vector<TrainingSample> samples;
};

constexpr uint32_t kDataMagic = 0x50535A41U;  // "AZSP"
constexpr uint32_t kDataVersion = 2U;

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void write_vector_raw(std::ofstream& out, const std::vector<T>& values) {
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(sizeof(T) * values.size()));
    }
}

void write_string(std::ofstream& out, const std::string& value) {
    const uint32_t size = static_cast<uint32_t>(value.size());
    write_pod(out, size);
    if (size > 0U) {
        out.write(value.data(), static_cast<std::streamsize>(size));
    }
}

[[nodiscard]] std::vector<uint8_t> pack_board_planes(const EncodedState& encoded) {
    std::vector<uint8_t> packed(encoded.board_planes.size(), 0U);
    for (size_t i = 0; i < encoded.board_planes.size(); ++i) {
        packed[i] = encoded.board_planes[i] > 0.5f ? 1U : 0U;
    }
    return packed;
}

[[nodiscard]] std::vector<int8_t> pack_last_move_markers(const EncodedState& encoded) {
    std::vector<int8_t> packed(encoded.last_move_markers.size(), 0);
    for (size_t i = 0; i < encoded.last_move_markers.size(); ++i) {
        const float value = encoded.last_move_markers[i];
        packed[i] = value > 0.5f ? static_cast<int8_t>(1) : (value < -0.5f ? static_cast<int8_t>(-1) : static_cast<int8_t>(0));
    }
    return packed;
}

[[nodiscard]] std::vector<int32_t> pack_int64_to_int32(const std::vector<int64_t>& values) {
    std::vector<int32_t> packed;
    packed.reserve(values.size());
    for (int64_t value : values) {
        packed.push_back(static_cast<int32_t>(value));
    }
    return packed;
}

[[nodiscard]] std::vector<int32_t> pack_action_i64(const std::vector<ActionEntry>& actions, auto selector) {
    std::vector<int32_t> packed;
    packed.reserve(actions.size());
    for (const auto& action : actions) {
        packed.push_back(static_cast<int32_t>(selector(action)));
    }
    return packed;
}

[[nodiscard]] std::vector<float> pack_action_float(const std::vector<ActionEntry>& actions, auto selector) {
    std::vector<float> packed;
    packed.reserve(actions.size());
    for (const auto& action : actions) {
        packed.push_back(static_cast<float>(selector(action)));
    }
    return packed;
}

[[nodiscard]] std::vector<uint8_t> pack_action_submit(const std::vector<ActionEntry>& actions) {
    std::vector<uint8_t> packed;
    packed.reserve(actions.size());
    for (const auto& action : actions) {
        packed.push_back(action.is_submit != 0 ? 1U : 0U);
    }
    return packed;
}

class BinarySampleWriter {
public:
    explicit BinarySampleWriter(const std::filesystem::path& path)
        : out_(path, std::ios::binary | std::ios::trunc) {
        if (!out_) {
            throw std::runtime_error("Could not open self-play output file: " + path.string());
        }
    }

    void write_header(uint32_t num_games) {
        write_pod(out_, kDataMagic);
        write_pod(out_, kDataVersion);
        write_pod(out_, num_games);
    }

    void write_game(const GameResult& game) {
        write_pod(out_, game.outcome);
        write_pod(out_, static_cast<int32_t>(game.total_semimoves));
        write_pod(out_, static_cast<int32_t>(game.board_limit));
        write_string(out_, game.terminal_reason);
        write_string(out_, game.pgn);

        write_pod(out_, static_cast<uint32_t>(game.move_history.size()));
        for (const auto& move : game.move_history) {
            write_pod(out_, static_cast<int8_t>(move.player));
            write_pod(out_, static_cast<uint8_t>(move.is_submit ? 1U : 0U));
            write_pod(out_, move.root_value);
            write_pod(out_, static_cast<int32_t>(move.board_count));
            write_string(out_, move.move_text);
        }

        write_pod(out_, static_cast<uint32_t>(game.samples.size()));
        for (const auto& sample : game.samples) {
            write_pod(out_, static_cast<int8_t>(sample.player));
            write_pod(out_, sample.urgency);
            write_pod(out_, sample.value_target);
            write_pod(out_, static_cast<int32_t>(sample.encoded_state.num_boards));
            write_pod(out_, static_cast<uint32_t>(sample.action_entries.size()));

            const std::vector<uint8_t> board_planes = pack_board_planes(sample.encoded_state);
            const std::vector<int8_t> last_move_markers = pack_last_move_markers(sample.encoded_state);
            const std::vector<int32_t> l_coords = pack_int64_to_int32(sample.encoded_state.l_coords);
            const std::vector<int32_t> t_coords = pack_int64_to_int32(sample.encoded_state.t_coords);
            const std::vector<int32_t> action_board_indices = pack_action_i64(sample.action_entries, [](const ActionEntry& e) { return e.board_idx; });
            const std::vector<int32_t> action_from_squares = pack_action_i64(sample.action_entries, [](const ActionEntry& e) { return e.from_sq; });
            const std::vector<int32_t> action_to_squares = pack_action_i64(sample.action_entries, [](const ActionEntry& e) { return e.to_sq; });
            const std::vector<float> action_delta_t = pack_action_float(sample.action_entries, [](const ActionEntry& e) { return e.delta_t; });
            const std::vector<float> action_delta_l = pack_action_float(sample.action_entries, [](const ActionEntry& e) { return e.delta_l; });
            const std::vector<uint8_t> action_is_submit = pack_action_submit(sample.action_entries);

            write_vector_raw(out_, board_planes);
            write_vector_raw(out_, last_move_markers);
            write_vector_raw(out_, l_coords);
            write_vector_raw(out_, t_coords);
            write_vector_raw(out_, sample.policy_probs);
            write_vector_raw(out_, action_board_indices);
            write_vector_raw(out_, action_from_squares);
            write_vector_raw(out_, action_to_squares);
            write_vector_raw(out_, action_delta_t);
            write_vector_raw(out_, action_delta_l);
            write_vector_raw(out_, action_is_submit);
        }
    }

private:
    std::ofstream out_;
};

[[nodiscard]] bool is_royal_piece(piece_t piece) {
    const piece_t p = piece_name(piece);
    return p == KING_W || p == KING_B
        || p == KING_UW || p == KING_UB
        || p == COMMON_KING_W || p == COMMON_KING_B
        || p == ROYAL_QUEEN_W || p == ROYAL_QUEEN_B;
}

[[nodiscard]] int piece_value(piece_t p) {
    switch (to_white(piece_name(p))) {
        case KING_W: return 0;
        case QUEEN_W: return 9;
        case ROOK_W: return 5;
        case BISHOP_W: return 3;
        case KNIGHT_W: return 3;
        case PAWN_W: return 1;
        case UNICORN_W: return 3;
        case DRAGON_W: return 5;
        case BRAWN_W: return 1;
        case PRINCESS_W: return 7;
        case ROYAL_QUEEN_W: return 9;
        case COMMON_KING_W: return 3;
        default: return 0;
    }
}

[[nodiscard]] float white_to_current_player_value(float outcome_white, int current_player) {
    return current_player == 0 ? outcome_white : -outcome_white;
}

[[nodiscard]] float no_legal_action_terminal_value() {
    return -1.0f;
}

[[nodiscard]] std::string outcome_text(float outcome) {
    if (outcome == 1.0f) {
        return "White wins";
    }
    if (outcome == -1.0f) {
        return "Black wins";
    }
    if (outcome == 0.0f) {
        return "Draw";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << outcome;
    return oss.str();
}

void apply_variant_preset(SearchConfig& cfg, const std::string& variant_name) {
    if (variant_name == "very_small") {
        cfg.variant_name = variant_name;
        cfg.variant_pgn = "[Board \"Very Small - Open\"]\n[Mode \"5D\"]\n";
        cfg.board_side = 4;
        cfg.board_squares = 16;
        cfg.piece_channels = kDefaultPieceChannels;
        cfg.line_shift = kDefaultLineShift;
        return;
    }
    if (variant_name == "standard") {
        cfg.variant_name = variant_name;
        cfg.variant_pgn = "[Board \"Standard\"]\n[Mode \"5D\"]\n";
        cfg.board_side = 8;
        cfg.board_squares = 64;
        cfg.piece_channels = kDefaultPieceChannels;
        cfg.line_shift = kDefaultLineShift;
        return;
    }
    if (variant_name == "standard_turn_zero") {
        cfg.variant_name = variant_name;
        cfg.variant_pgn = "[Board \"Standard - Turn Zero\"]\n[Mode \"5D\"]\n";
        cfg.board_side = 8;
        cfg.board_squares = 64;
        cfg.piece_channels = kDefaultPieceChannels;
        cfg.line_shift = kDefaultLineShift;
        return;
    }
    throw std::runtime_error("Unknown variant: " + variant_name);
}

[[nodiscard]] std::vector<float> sample_dirichlet(std::mt19937& rng, int size, float alpha) {
    std::vector<float> noise(size, 0.0f);
    if (size <= 0) {
        return noise;
    }
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float sum = 0.0f;
    for (float& value : noise) {
        value = gamma(rng);
        sum += value;
    }
    if (sum <= 0.0f) {
        const float uniform = 1.0f / static_cast<float>(size);
        std::fill(noise.begin(), noise.end(), uniform);
        return noise;
    }
    for (float& value : noise) {
        value /= sum;
    }
    return noise;
}

class CaptureKingEnv {
public:
    CaptureKingEnv(std::string variant_pgn, int board_limit, int board_side, int board_squares, int line_shift, float material_scale)
        : variant_pgn_(std::move(variant_pgn)),
          board_limit_(board_limit),
          board_side_(board_side),
          board_squares_(board_squares),
          line_shift_(line_shift),
          material_scale_(material_scale),
          game_(game::from_pgn(variant_pgn_)) {
        reset();
    }

    void reset() {
        game_ = game::from_pgn(variant_pgn_);
        pending_semimoves_.clear();
        turn_history_.clear();
        last_semimove_.reset();
        done_ = false;
        outcome_ = 0.0f;
        terminal_reason_.clear();
        total_semimoves_ = 0;
    }

    [[nodiscard]] int current_player() const {
        const auto [present_t, is_black] = game_.get_current_present();
        (void)present_t;
        return is_black ? 1 : 0;
    }

    [[nodiscard]] const state& state_ref() const {
        return game_.get_current_state();
    }

    [[nodiscard]] bool done() const {
        return done_;
    }

    [[nodiscard]] float outcome() const {
        return outcome_;
    }

    [[nodiscard]] int board_count() const {
        return static_cast<int>(state_ref().get_boards().size());
    }

    [[nodiscard]] int total_semimoves() const {
        return total_semimoves_;
    }

    [[nodiscard]] CaptureKingEnv clone() const {
        CaptureKingEnv clone(variant_pgn_, board_limit_, board_side_, board_squares_, line_shift_, material_scale_);
        clone.done_ = done_;
        clone.outcome_ = outcome_;
        clone.terminal_reason_ = terminal_reason_;
        clone.total_semimoves_ = total_semimoves_;

        for (const auto& turn : turn_history_) {
            for (const auto& sm : turn) {
                if (!clone.game_.apply_move_unsafe(sm.to_ext_move())) {
                    throw std::runtime_error("Failed to replay completed semimove while cloning.");
                }
            }
            if (!clone.game_.submit_unsafe()) {
                throw std::runtime_error("Failed to replay submit while cloning.");
            }
        }
        for (const auto& sm : pending_semimoves_) {
            if (!clone.game_.apply_move_unsafe(sm.to_ext_move())) {
                throw std::runtime_error("Failed to replay pending semimove while cloning.");
            }
        }

        clone.turn_history_ = turn_history_;
        clone.pending_semimoves_ = pending_semimoves_;
        clone.last_semimove_ = last_semimove_;
        return clone;
    }

    [[nodiscard]] std::pair<std::vector<Semimove>, bool> get_legal_frontier() const {
        if (done_) {
            return {{}, false};
        }

        std::vector<Semimove> semimoves;
        std::set<std::tuple<int, int, int, int, int, int, int, int>> seen;
        const auto [mandatory, optional_lines, unplayable] = state_ref().get_timeline_status();
        (void)unplayable;

        auto process_lines = [&](const std::vector<int>& lines) {
            for (int line : lines) {
                const auto [t, c] = state_ref().get_timeline_end(line);
                auto board_ptr = state_ref().get_board(line, t, c);
                if (!board_ptr) {
                    continue;
                }
                const bitboard_t pieces = c
                    ? (board_ptr->black() & ~board_ptr->white())
                    : (board_ptr->white() & ~board_ptr->black());
                for (int pos : marked_pos(pieces)) {
                    const vec4 from(pos, vec4(0, 0, t, line));
                    for (const vec4& to : state_ref().gen_piece_move_unsafe(from, c)) {
                        Semimove sm{line, from, to};
                        const auto key = std::make_tuple(
                            from.x(), from.y(), from.t(), from.l(),
                            to.x(), to.y(), to.t(), to.l()
                        );
                        if (seen.insert(key).second) {
                            semimoves.push_back(sm);
                        }
                    }
                }
            }
        };

        process_lines(mandatory);
        process_lines(optional_lines);

        return {apply_lexicographic_filter(semimoves), mandatory.empty()};
    }

    [[nodiscard]] bool apply_semimove(const Semimove& sm, bool validate = true) {
        if (done_) {
            return false;
        }
        if (validate) {
            const auto [frontier, can_submit] = get_legal_frontier();
            (void)can_submit;
            if (std::find(frontier.begin(), frontier.end(), sm) == frontier.end()) {
                return false;
            }
        }

        const std::optional<float> capture_outcome = capture_royal_outcome(sm);
        if (!game_.apply_move_unsafe(sm.to_ext_move())) {
            return false;
        }
        pending_semimoves_.push_back(sm);
        last_semimove_ = sm;
        total_semimoves_ += 1;
        if (capture_outcome.has_value()) {
            done_ = true;
            outcome_ = *capture_outcome;
            terminal_reason_ = "capture_king";
        }
        return true;
    }

    [[nodiscard]] std::optional<float> submit_turn(bool assume_legal = false) {
        if (done_) {
            return outcome_;
        }
        if (!assume_legal) {
            const auto [frontier, can_submit] = get_legal_frontier();
            (void)frontier;
            if (!can_submit) {
                return std::nullopt;
            }
        }
        if (!game_.submit_unsafe()) {
            return std::nullopt;
        }

        turn_history_.push_back(pending_semimoves_);
        pending_semimoves_.clear();
        last_semimove_.reset();
        return check_terminal();
    }

    [[nodiscard]] EncodedState encode_state() const {
        EncodedState encoded;
        const auto boards_info = state_ref().get_boards();
        encoded.num_boards = static_cast<int>(boards_info.size());
        encoded.board_planes.assign(
            static_cast<size_t>(encoded.num_boards) * kDefaultPieceChannels * board_squares_,
            0.0f
        );
        encoded.last_move_markers.assign(
            static_cast<size_t>(encoded.num_boards) * board_squares_,
            0.0f
        );
        encoded.l_coords.resize(encoded.num_boards);
        encoded.t_coords.resize(encoded.num_boards);
        encoded.board_keys.reserve(encoded.num_boards);

        for (int i = 0; i < encoded.num_boards; ++i) {
            const auto& [l, t, c, fen] = boards_info[static_cast<size_t>(i)];
            (void)fen;
            encoded.l_coords[static_cast<size_t>(i)] = static_cast<int64_t>(l + line_shift_);
            encoded.t_coords[static_cast<size_t>(i)] = static_cast<int64_t>(t);
            encoded.board_keys.push_back(BoardKey{l, t, c});

            auto board_ptr = state_ref().get_board(l, t, c);
            if (!board_ptr) {
                continue;
            }

            for (int y = 0; y < board_side_; ++y) {
                for (int x = 0; x < board_side_; ++x) {
                    const int sq = x + y * board_side_;
                    const int board_pos = ppos(x, y);
                    const piece_t raw_piece = board_ptr->get_piece(board_pos);
                    const piece_t piece = piece_name(raw_piece);

                    const size_t plane_base =
                        (static_cast<size_t>(i) * kDefaultPieceChannels * board_squares_) + static_cast<size_t>(sq);
                    if (raw_piece != NO_PIECE && raw_piece != WALL_PIECE) {
                        const int channel = piece_channel(piece);
                        if (channel >= 0) {
                            encoded.board_planes[plane_base + static_cast<size_t>(channel) * board_squares_] = 1.0f;
                        }
                        encoded.board_planes[plane_base + 13ULL * board_squares_] = 1.0f;
                    }
                    if (board_ptr->umove() & pmask(board_pos)) {
                        encoded.board_planes[plane_base + 12ULL * board_squares_] = 1.0f;
                    }
                }
            }
        }

        if (last_semimove_.has_value()) {
            const Semimove& sm = *last_semimove_;
            const bool player_color = current_player() == 1;
            for (int i = 0; i < encoded.num_boards; ++i) {
                const auto& key = encoded.board_keys[static_cast<size_t>(i)];
                if (key.l == sm.from.l() && key.t == sm.from.t() && key.c == player_color) {
                    const int pos = sm.from.x() + sm.from.y() * board_side_;
                    if (0 <= pos && pos < board_squares_) {
                        encoded.last_move_markers[static_cast<size_t>(i) * board_squares_ + static_cast<size_t>(pos)] = 1.0f;
                    }
                }
                if (key.l == sm.to.l() && key.t == sm.to.t() && key.c == player_color) {
                    const int pos = sm.to.x() + sm.to.y() * board_side_;
                    if (0 <= pos && pos < board_squares_) {
                        encoded.last_move_markers[static_cast<size_t>(i) * board_squares_ + static_cast<size_t>(pos)] = -1.0f;
                    }
                }
            }
        }

        return encoded;
    }

    [[nodiscard]] std::string format_move(const Semimove& sm) const {
        return state_ref().pretty_move<state::SHOW_CAPTURE | state::SHOW_PROMOTION>(full_move(sm.from, sm.to));
    }

    [[nodiscard]] std::string transposition_key() const {
        std::ostringstream oss;
        oss << state_ref().show_fen()
            << "|p=" << current_player()
            << "|done=" << (done_ ? 1 : 0);
        if (done_) {
            oss << "|out=" << outcome_;
        }
        if (last_semimove_.has_value()) {
            const Semimove& sm = *last_semimove_;
            oss << "|last="
                << sm.line_idx << ':'
                << sm.from.x() << ',' << sm.from.y() << ',' << sm.from.t() << ',' << sm.from.l() << ':'
                << sm.to.x() << ',' << sm.to.y() << ',' << sm.to.t() << ',' << sm.to.l();
        } else {
            oss << "|last=-";
        }
        return oss.str();
    }

    [[nodiscard]] std::string show_pgn(uint16_t show_flags = state::SHOW_CAPTURE | state::SHOW_PROMOTION) {
        return game_.show_pgn(show_flags);
    }

private:
    [[nodiscard]] static int piece_channel(piece_t piece) {
        switch (piece) {
            case KING_W: return 0;
            case QUEEN_W: return 1;
            case ROOK_W: return 2;
            case BISHOP_W: return 3;
            case KNIGHT_W: return 4;
            case PAWN_W: return 5;
            case KING_B: return 6;
            case QUEEN_B: return 7;
            case ROOK_B: return 8;
            case BISHOP_B: return 9;
            case KNIGHT_B: return 10;
            case PAWN_B: return 11;
            default: return -1;
        }
    }

    [[nodiscard]] std::vector<Semimove> apply_lexicographic_filter(
        const std::vector<Semimove>& semimoves
    ) const {
        if (!last_semimove_.has_value()) {
            return semimoves;
        }
        std::vector<Semimove> filtered;
        filtered.reserve(semimoves.size());
        for (const auto& sm : semimoves) {
            if (semimoves_are_commutable(sm, *last_semimove_) && sm.sort_key() < last_semimove_->sort_key()) {
                continue;
            }
            filtered.push_back(sm);
        }
        return filtered;
    }

    [[nodiscard]] std::optional<float> capture_royal_outcome(const Semimove& sm) const {
        const bool mover_is_black = current_player() == 1;
        const piece_t piece = state_ref().get_piece(sm.to, mover_is_black);
        if (piece == NO_PIECE || piece == WALL_PIECE) {
            return std::nullopt;
        }
        if (!is_royal_piece(piece)) {
            return std::nullopt;
        }
        return mover_is_black ? std::optional<float>(-1.0f) : std::optional<float>(1.0f);
    }

    [[nodiscard]] std::pair<int, int> material_count() const {
        int white_material = 0;
        int black_material = 0;
        for (const auto& [l, t, c, fen] : state_ref().get_boards()) {
            (void)fen;
            auto board_ptr = state_ref().get_board(l, t, c);
            if (!board_ptr) {
                continue;
            }
            for (int pos = 0; pos < BOARD_SIZE; ++pos) {
                const piece_t p = board_ptr->get_piece(pos);
                if (p == NO_PIECE || p == WALL_PIECE) {
                    continue;
                }
                const int value = piece_value(p);
                if (piece_color(piece_name(p))) {
                    black_material += value;
                } else {
                    white_material += value;
                }
            }
        }
        return {white_material, black_material};
    }

    [[nodiscard]] float material_score() const {
        const auto [white_material, black_material] = material_count();
        const float total = static_cast<float>(white_material + black_material) + 1e-8f;
        const float diff = static_cast<float>(white_material - black_material);
        return std::tanh(diff / total * material_scale_);
    }

    [[nodiscard]] std::optional<float> check_terminal() {
        if (done_) {
            return outcome_;
        }
        if (board_count() >= board_limit_) {
            done_ = true;
            outcome_ = material_score();
            terminal_reason_ = "material";
            return outcome_;
        }
        return std::nullopt;
    }

public:
    [[nodiscard]] const std::string& terminal_reason() const {
        return terminal_reason_;
    }

private:
    [[nodiscard]] bool is_playable_board_coord(int t, int l) const {
        const auto [mandatory, optional_lines, unplayable] = state_ref().get_timeline_status();
        (void)unplayable;
        const bool playable_line =
            std::find(mandatory.begin(), mandatory.end(), l) != mandatory.end() ||
            std::find(optional_lines.begin(), optional_lines.end(), l) != optional_lines.end();
        if (!playable_line) {
            return false;
        }
        const auto [present_t, present_player] = state_ref().get_present();
        (void)present_t;
        const auto [end_t, end_player] = state_ref().get_timeline_end(l);
        return end_t == t && end_player == present_player;
    }

    [[nodiscard]] bool moves_to_inactive_board(const Semimove& sm) const {
        return !is_playable_board_coord(sm.to.t(), sm.to.l());
    }

    [[nodiscard]] static bool destination_hits_other_source(const Semimove& lhs, const Semimove& rhs) {
        return std::make_pair(lhs.to.t(), lhs.to.l()) == std::make_pair(rhs.from.t(), rhs.from.l()) ||
               std::make_pair(rhs.to.t(), rhs.to.l()) == std::make_pair(lhs.from.t(), lhs.from.l());
    }

    [[nodiscard]] bool semimoves_are_commutable(const Semimove& lhs, const Semimove& rhs) const {
        if (destination_hits_other_source(lhs, rhs)) {
            return false;
        }
        return !(moves_to_inactive_board(lhs) && moves_to_inactive_board(rhs));
    }

    std::string variant_pgn_;
    int board_limit_ = 25;
    int board_side_ = 4;
    int board_squares_ = 16;
    int line_shift_ = kDefaultLineShift;
    float material_scale_ = 2.0f;
    game game_;
    std::vector<std::vector<Semimove>> turn_history_;
    std::vector<Semimove> pending_semimoves_;
    std::optional<Semimove> last_semimove_;
    bool done_ = false;
    float outcome_ = 0.0f;
    std::string terminal_reason_;
    int total_semimoves_ = 0;
};

class OnnxPolicyValue {
public:
    struct BatchRequest {
        EncodedState encoded;
        std::vector<ActionEntry> actions;
        float urgency = 0.0f;
    };

    OnnxPolicyValue(
        const std::filesystem::path& model_path,
        const std::string& provider,
        int cuda_device_id,
        int intra_threads,
        int piece_channels,
        int board_squares,
        int batch_slots,
        RunnerProfile* profile = nullptr
    )
        : env_(ORT_LOGGING_LEVEL_ERROR, "az_selfplay_onnx"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPUInput)),
          provider_(provider),
          piece_channels_(piece_channels),
          board_squares_(board_squares),
          batch_slots_(std::max(1, batch_slots)),
          profile_(profile) {
        diag_log("OnnxPolicyValue ctor begin provider=" + provider_ + " model=" + model_path.string());
        session_options_.SetIntraOpNumThreads(intra_threads);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        if (provider_ == "cuda") {
            diag_log("Appending CUDA execution provider");
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = cuda_device_id;
            cuda_options.do_copy_in_default_stream = 1;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
        } else if (provider_ != "cpu") {
            throw std::runtime_error("Unsupported ONNX execution provider: " + provider_);
        }
        diag_log("Creating ORT session");
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        diag_log("ORT session created");
    }

    [[nodiscard]] std::pair<float, std::vector<float>> predict_actions(
        const EncodedState& encoded,
        const std::vector<ActionEntry>& actions,
        float urgency
    ) const {
        BatchRequest request{encoded, actions, urgency};
        auto outputs = predict_actions_batch({request});
        return outputs.empty() ? std::pair<float, std::vector<float>>{0.0f, {}} : std::move(outputs.front());
    }

    [[nodiscard]] std::vector<std::pair<float, std::vector<float>>> predict_actions_batch(
        const std::vector<BatchRequest>& requests
    ) const {
        const auto t0 = std::chrono::steady_clock::now();
        if (requests.empty()) {
            if (profile_ != nullptr) {
                profile_->onnx_predict_actions.add(seconds_since(t0));
                profile_->network_calls += 1;
            }
            return {};
        }
        size_t max_boards = 0;
        size_t total_actions = 0;
        for (const auto& request : requests) {
            max_boards = std::max(max_boards, static_cast<size_t>(request.encoded.num_boards));
            total_actions += request.actions.size();
        }
        if (max_boards == 0) {
            max_boards = 1;
        }
        if (requests.size() > static_cast<size_t>(batch_slots_)) {
            throw std::runtime_error("predict_actions_batch received more requests than ONNX batch slots.");
        }

        const size_t batch_size = static_cast<size_t>(batch_slots_);
        const size_t active_batch = requests.size();
        std::vector<float> board_planes(batch_size * max_boards * static_cast<size_t>(piece_channels_) * static_cast<size_t>(board_squares_), 0.0f);
        std::vector<float> last_move_markers(batch_size * max_boards * static_cast<size_t>(board_squares_), 0.0f);
        std::vector<int64_t> l_coords(batch_size * max_boards, 0);
        std::vector<int64_t> t_coords(batch_size * max_boards, 0);
        std::vector<int64_t> used_board_counts(batch_size, 0);
        std::vector<float> urgency_values(batch_size, 0.0f);
        std::vector<int64_t> action_state_indices;
        std::vector<int64_t> action_board_indices;
        std::vector<int64_t> action_from_squares;
        std::vector<int64_t> action_to_squares;
        std::vector<float> action_delta_t;
        std::vector<float> action_delta_l;
        std::vector<int64_t> action_is_submit;
        std::vector<size_t> action_counts;
        action_state_indices.reserve(total_actions);
        action_board_indices.reserve(total_actions);
        action_from_squares.reserve(total_actions);
        action_to_squares.reserve(total_actions);
        action_delta_t.reserve(total_actions);
        action_delta_l.reserve(total_actions);
        action_is_submit.reserve(total_actions);
        action_counts.reserve(active_batch);

        const size_t board_plane_stride = max_boards * static_cast<size_t>(piece_channels_) * static_cast<size_t>(board_squares_);
        const size_t marker_stride = max_boards * static_cast<size_t>(board_squares_);
        const size_t coord_stride = max_boards;
        for (size_t batch_idx = 0; batch_idx < active_batch; ++batch_idx) {
            const auto& request = requests[batch_idx];
            const EncodedState& encoded = request.encoded;
            const size_t num_boards = static_cast<size_t>(encoded.num_boards);
            used_board_counts[batch_idx] = static_cast<int64_t>(encoded.num_boards);
            urgency_values[batch_idx] = request.urgency;
            if (num_boards > 0) {
                std::copy(
                    encoded.board_planes.begin(),
                    encoded.board_planes.end(),
                    board_planes.begin() + static_cast<std::ptrdiff_t>(batch_idx * board_plane_stride)
                );
                std::copy(
                    encoded.last_move_markers.begin(),
                    encoded.last_move_markers.end(),
                    last_move_markers.begin() + static_cast<std::ptrdiff_t>(batch_idx * marker_stride)
                );
                std::copy(
                    encoded.l_coords.begin(),
                    encoded.l_coords.end(),
                    l_coords.begin() + static_cast<std::ptrdiff_t>(batch_idx * coord_stride)
                );
                std::copy(
                    encoded.t_coords.begin(),
                    encoded.t_coords.end(),
                    t_coords.begin() + static_cast<std::ptrdiff_t>(batch_idx * coord_stride)
                );
            }
            action_counts.push_back(request.actions.size());
            for (const auto& action : request.actions) {
                action_state_indices.push_back(static_cast<int64_t>(batch_idx));
                action_board_indices.push_back(action.board_idx);
                action_from_squares.push_back(action.from_sq);
                action_to_squares.push_back(action.to_sq);
                action_delta_t.push_back(action.delta_t);
                action_delta_l.push_back(action.delta_l);
                action_is_submit.push_back(action.is_submit);
            }
        }

        diag_log(
            "predict_actions_batch begin batch=" + std::to_string(active_batch) +
            "/" + std::to_string(batch_slots_) +
            " max_boards=" + std::to_string(max_boards) +
            " actions=" + std::to_string(total_actions) +
            " provider=" + provider_
        );

        const std::array<int64_t, 4> board_planes_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(max_boards),
            static_cast<int64_t>(piece_channels_),
            static_cast<int64_t>(board_squares_)
        };
        const std::array<int64_t, 3> marker_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(max_boards),
            static_cast<int64_t>(board_squares_)
        };
        const std::array<int64_t, 2> board_dim = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(max_boards)
        };
        const std::array<int64_t, 1> used_board_counts_shape = {static_cast<int64_t>(batch_size)};
        const std::array<int64_t, 1> urgency_shape = {static_cast<int64_t>(batch_size)};
        const std::array<int64_t, 1> action_dim = {static_cast<int64_t>(action_state_indices.size())};

        std::array<Ort::Value, 13> input_tensors = {
            Ort::Value::CreateTensor<float>(memory_info_, board_planes.data(), board_planes.size(), board_planes_shape.data(), board_planes_shape.size()),
            Ort::Value::CreateTensor<float>(memory_info_, last_move_markers.data(), last_move_markers.size(), marker_shape.data(), marker_shape.size()),
            Ort::Value::CreateTensor<int64_t>(memory_info_, l_coords.data(), l_coords.size(), board_dim.data(), board_dim.size()),
            Ort::Value::CreateTensor<int64_t>(memory_info_, t_coords.data(), t_coords.size(), board_dim.data(), board_dim.size()),
            Ort::Value::CreateTensor<int64_t>(memory_info_, used_board_counts.data(), used_board_counts.size(), used_board_counts_shape.data(), used_board_counts_shape.size()),
            Ort::Value::CreateTensor<float>(memory_info_, urgency_values.data(), urgency_values.size(), urgency_shape.data(), urgency_shape.size()),
            Ort::Value::CreateTensor<int64_t>(memory_info_, action_state_indices.data(), action_state_indices.size(), action_dim.data(), action_dim.size()),
            Ort::Value::CreateTensor<int64_t>(memory_info_, action_board_indices.data(), action_board_indices.size(), action_dim.data(), action_dim.size()),
            Ort::Value::CreateTensor<int64_t>(memory_info_, action_from_squares.data(), action_from_squares.size(), action_dim.data(), action_dim.size()),
            Ort::Value::CreateTensor<int64_t>(memory_info_, action_to_squares.data(), action_to_squares.size(), action_dim.data(), action_dim.size()),
            Ort::Value::CreateTensor<float>(memory_info_, action_delta_t.data(), action_delta_t.size(), action_dim.data(), action_dim.size()),
            Ort::Value::CreateTensor<float>(memory_info_, action_delta_l.data(), action_delta_l.size(), action_dim.data(), action_dim.size()),
            Ort::Value::CreateTensor<int64_t>(memory_info_, action_is_submit.data(), action_is_submit.size(), action_dim.data(), action_dim.size()),
        };

        diag_log("session.Run begin");
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_.data(),
            output_names_.size()
        );
        diag_log("session.Run end");

        const float* values_ptr = outputs[0].GetTensorData<float>();
        diag_log("read value output");
        auto type_info = outputs[1].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = type_info.GetShape();
        size_t output_len = 1;
        for (int64_t dim : output_shape) {
            output_len *= static_cast<size_t>(dim);
        }
        const float* logits_ptr = outputs[1].GetTensorData<float>();
        std::vector<std::pair<float, std::vector<float>>> result;
        result.reserve(active_batch);
        size_t offset = 0;
        for (size_t batch_idx = 0; batch_idx < active_batch; ++batch_idx) {
            const size_t count = action_counts[batch_idx];
            result.emplace_back(
                values_ptr[batch_idx],
                std::vector<float>(logits_ptr + offset, logits_ptr + offset + count)
            );
            offset += count;
        }
        diag_log("predict_actions_batch end logits=" + std::to_string(output_len));
        if (profile_ != nullptr) {
            profile_->onnx_predict_actions.add(seconds_since(t0));
            profile_->network_calls += static_cast<uint64_t>(active_batch);
        }
        return result;
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;
    std::unique_ptr<Ort::Session> session_;
    std::string provider_ = "cpu";
    int piece_channels_ = kDefaultPieceChannels;
    int board_squares_ = 16;
    int batch_slots_ = 1;
    RunnerProfile* profile_ = nullptr;
    const std::array<const char*, 13> input_names_ = {
        "board_planes",
        "last_move_markers",
        "l_coords",
        "t_coords",
        "used_board_counts",
        "urgency",
        "action_state_indices",
        "action_board_indices",
        "action_from_squares",
        "action_to_squares",
        "action_delta_t",
        "action_delta_l",
        "action_is_submit",
    };
    const std::array<const char*, 2> output_names_ = {"value", "action_logits"};
};

class MCTSNode {
public:
    explicit MCTSNode(MCTSNode* parent = nullptr, ActionChoice action = {})
        : parent(parent), action(action) {}

    [[nodiscard]] float q_value() const {
        if (visit_count == 0) {
            return 0.0f;
        }
        return value_sum / static_cast<float>(visit_count);
    }

    [[nodiscard]] float ucb_score(float c_puct) const {
        const int parent_visits = parent ? parent->visit_count : 1;
        const float u = c_puct * prior * std::sqrt(static_cast<float>(parent_visits)) /
            (1.0f + static_cast<float>(visit_count));
        return q_value() + u;
    }

    [[nodiscard]] MCTSNode* select_child(float c_puct) {
        MCTSNode* best = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();
        for (auto& child : children) {
            const float score = child->ucb_score(c_puct);
            if (score > best_score) {
                best_score = score;
                best = child.get();
            }
        }
        return best;
    }

    void expand(const std::vector<std::pair<ActionChoice, float>>& child_specs, bool terminal, float terminal_score) {
        is_expanded = true;
        is_terminal = terminal;
        this->terminal_value = terminal_score;
        if (terminal) {
            return;
        }
        children.reserve(child_specs.size());
        for (const auto& [child_action, child_prior] : child_specs) {
            auto child = std::make_unique<MCTSNode>(this, child_action);
            child->prior = child_prior;
            children.push_back(std::move(child));
        }
    }

    MCTSNode* parent = nullptr;
    ActionChoice action{};
    float prior = 0.0f;
    int visit_count = 0;
    float value_sum = 0.0f;
    bool is_expanded = false;
    bool is_terminal = false;
    float terminal_value = 0.0f;
    std::vector<std::unique_ptr<MCTSNode>> children;
};

class SemimoveMcts {
public:
    struct TranspositionEntry {
        float value = 0.0f;
        bool terminal = false;
        float terminal_value = 0.0f;
        std::vector<std::pair<ActionChoice, float>> child_specs;
        std::vector<ActionEntry> action_entries;
    };

    SemimoveMcts(const SearchConfig& cfg, OnnxPolicyValue& network, std::mt19937& rng, RunnerProfile* profile = nullptr)
        : cfg_(cfg), network_(network), rng_(rng), profile_(profile) {}

    struct SearchResult {
        ActionChoice action;
        float root_value = 0.0f;
        std::vector<float> policy_probs;
        std::vector<ActionEntry> action_entries;
    };

    struct PendingLeaf {
        MCTSNode* node = nullptr;
        CaptureKingEnv env;
        std::vector<MCTSNode*> path;
        std::chrono::steady_clock::time_point sim_start{};

        PendingLeaf(
            MCTSNode* node_in,
            CaptureKingEnv env_in,
            std::vector<MCTSNode*> path_in,
            std::chrono::steady_clock::time_point sim_start_in
        )
            : node(node_in), env(std::move(env_in)), path(std::move(path_in)), sim_start(sim_start_in) {}
    };

    [[nodiscard]] SearchResult select_action(CaptureKingEnv& env, float urgency, float temperature) {
        const auto select_start = std::chrono::steady_clock::now();
        transposition_table_.clear();
        MCTSNode root;
        std::vector<ActionEntry> root_entries;
        (void)expand_node(root, env, urgency, &root_entries);

        if (!root.children.empty() && cfg_.dirichlet_epsilon > 0.0f) {
            const std::vector<float> noise = sample_dirichlet(
                rng_,
                static_cast<int>(root.children.size()),
                cfg_.dirichlet_alpha
            );
            for (size_t i = 0; i < root.children.size(); ++i) {
                root.children[i]->prior =
                    (1.0f - cfg_.dirichlet_epsilon) * root.children[i]->prior +
                    cfg_.dirichlet_epsilon * noise[i];
            }
        }

        int sim = 0;
        while (sim < cfg_.num_simulations) {
            std::vector<PendingLeaf> pending;
            pending.reserve(static_cast<size_t>(std::max(1, cfg_.leaf_batch_size)));

            while (sim < cfg_.num_simulations && static_cast<int>(pending.size()) < std::max(1, cfg_.leaf_batch_size)) {
                const auto sim_start = std::chrono::steady_clock::now();
                CaptureKingEnv scratch = env.clone();
                if (profile_ != nullptr) {
                    profile_->env_clone.add(seconds_since(sim_start));
                    profile_->simulations += 1;
                }
                std::vector<MCTSNode*> path;
                MCTSNode* node = &root;
                path.push_back(node);

                const auto tree_select_start = std::chrono::steady_clock::now();
                while (node->is_expanded && !node->is_terminal && !node->children.empty()) {
                    node = node->select_child(cfg_.c_puct);
                    path.push_back(node);

                    if (node->action.is_submit) {
                        const auto submit_start = std::chrono::steady_clock::now();
                        const auto outcome = scratch.submit_turn(true);
                        if (profile_ != nullptr) {
                            profile_->replay_submit_turn.add(seconds_since(submit_start));
                        }
                        if (outcome.has_value()) {
                            node->is_terminal = true;
                            node->terminal_value = white_to_current_player_value(*outcome, scratch.current_player());
                            break;
                        }
                    } else {
                        const auto apply_start = std::chrono::steady_clock::now();
                        if (!scratch.apply_semimove(node->action.semimove, false)) {
                            throw std::runtime_error("Expanded semimove failed during MCTS replay.");
                        }
                        if (profile_ != nullptr) {
                            profile_->replay_apply_semimove.add(seconds_since(apply_start));
                        }
                    }
                }
                if (profile_ != nullptr) {
                    profile_->tree_select.add(seconds_since(tree_select_start));
                }

                if (node->is_terminal) {
                    backpropagate(path, node->terminal_value);
                    if (profile_ != nullptr) {
                        profile_->mcts_simulation.add(seconds_since(sim_start));
                    }
                    sim += 1;
                    continue;
                }

                bool duplicate_pending = false;
                for (const auto& leaf : pending) {
                    if (leaf.node == node) {
                        duplicate_pending = true;
                        break;
                    }
                }
                if (duplicate_pending) {
                    break;
                }

                apply_virtual_visit(path);
                pending.emplace_back(node, std::move(scratch), std::move(path), sim_start);
                sim += 1;
            }

            if (!pending.empty()) {
                expand_pending_leaves_batched(pending, urgency);
            }
        }

        if (root.children.empty()) {
            if (profile_ != nullptr) {
                profile_->mcts_select_action.add(seconds_since(select_start));
            }
            return {ActionChoice{}, root.q_value(), {}};
        }

        std::vector<float> visits(root.children.size(), 0.0f);
        for (size_t i = 0; i < root.children.size(); ++i) {
            visits[i] = static_cast<float>(root.children[i]->visit_count);
        }
        std::vector<float> policy = visit_policy(visits, temperature);
        const auto sample_start = std::chrono::steady_clock::now();
        const size_t idx = sample_policy(policy);
        if (profile_ != nullptr) {
            profile_->sample_policy.add(seconds_since(sample_start));
            profile_->mcts_select_action.add(seconds_since(select_start));
        }
        return {root.children[idx]->action, root.q_value(), policy, std::move(root_entries)};
    }

private:
    void apply_virtual_visit(const std::vector<MCTSNode*>& path) {
        for (MCTSNode* path_node : path) {
            path_node->visit_count += 1;
        }
    }

    void accumulate_value_only(const std::vector<MCTSNode*>& path, float value) {
        const auto backprop_start = std::chrono::steady_clock::now();
        float running = value;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            MCTSNode* path_node = *it;
            path_node->value_sum += running;
            if (path_node->action.is_submit) {
                running = -running;
            }
        }
        if (profile_ != nullptr) {
            profile_->mcts_backprop.add(seconds_since(backprop_start));
        }
    }

    void backpropagate(const std::vector<MCTSNode*>& path, float value) {
        apply_virtual_visit(path);
        accumulate_value_only(path, value);
    }

    void expand_pending_leaves_batched(std::vector<PendingLeaf>& pending, float urgency) {
        const auto batch_expand_start = std::chrono::steady_clock::now();
        std::vector<OnnxPolicyValue::BatchRequest> batch_requests;
        batch_requests.reserve(pending.size());
        std::vector<size_t> request_leaf_indices;
        request_leaf_indices.reserve(pending.size());
        std::vector<float> resolved_values(pending.size(), 0.0f);

        for (size_t i = 0; i < pending.size(); ++i) {
            auto& leaf = pending[i];
            MCTSNode& node = *leaf.node;

            std::string tt_key;
            if (cfg_.use_transposition_table) {
                const auto tt_lookup_start = std::chrono::steady_clock::now();
                tt_key = leaf.env.transposition_key();
                auto found = transposition_table_.find(tt_key);
                if (profile_ != nullptr) {
                    profile_->tt_lookup.add(seconds_since(tt_lookup_start));
                }
                if (found != transposition_table_.end()) {
                    node.expand(
                        found->second.child_specs,
                        found->second.terminal,
                        found->second.terminal_value
                    );
                    resolved_values[i] = found->second.value;
                    if (profile_ != nullptr) {
                        profile_->tt_hits += 1;
                    }
                    continue;
                }
                if (profile_ != nullptr) {
                    profile_->tt_misses += 1;
                }
            }

            if (leaf.env.done()) {
                const float terminal_value = white_to_current_player_value(leaf.env.outcome(), leaf.env.current_player());
                node.expand({}, true, terminal_value);
                resolved_values[i] = terminal_value;
                if (cfg_.use_transposition_table) {
                    transposition_table_[tt_key] = TranspositionEntry{
                        terminal_value,
                        true,
                        terminal_value,
                        {},
                        {}
                    };
                }
                continue;
            }

            const auto frontier_start = std::chrono::steady_clock::now();
            const auto [legal_semimoves, can_submit] = leaf.env.get_legal_frontier();
            if (profile_ != nullptr) {
                profile_->legal_frontier.add(seconds_since(frontier_start));
                profile_->frontier_nodes += 1;
            }
            if (legal_semimoves.empty() && !can_submit) {
                const float terminal_value = no_legal_action_terminal_value();
                node.expand({}, true, terminal_value);
                resolved_values[i] = terminal_value;
                if (cfg_.use_transposition_table) {
                    transposition_table_[tt_key] = TranspositionEntry{
                        terminal_value,
                        true,
                        terminal_value,
                        {},
                        {}
                    };
                }
                continue;
            }

            const auto encode_start = std::chrono::steady_clock::now();
            EncodedState encoded = leaf.env.encode_state();
            if (profile_ != nullptr) {
                profile_->encode_state.add(seconds_since(encode_start));
            }
            const auto build_start = std::chrono::steady_clock::now();
            std::vector<ActionEntry> entries = build_action_entries(leaf.env, encoded, legal_semimoves, can_submit);
            if (profile_ != nullptr) {
                profile_->build_action_entries.add(seconds_since(build_start));
            }
            batch_requests.push_back(OnnxPolicyValue::BatchRequest{
                std::move(encoded),
                std::move(entries),
                urgency,
            });
            request_leaf_indices.push_back(i);
        }

        if (!batch_requests.empty()) {
            const auto batch_outputs = network_.predict_actions_batch(batch_requests);
            for (size_t req_idx = 0; req_idx < batch_requests.size(); ++req_idx) {
                const size_t leaf_idx = request_leaf_indices[req_idx];
                auto& leaf = pending[leaf_idx];
                auto& node = *leaf.node;
                const auto& [value, logits] = batch_outputs[req_idx];
                auto& request = batch_requests[req_idx];
                const auto softmax_start = std::chrono::steady_clock::now();
                const std::vector<float> priors = softmax(logits);
                if (profile_ != nullptr) {
                    profile_->softmax.add(seconds_since(softmax_start));
                }

                std::vector<std::pair<ActionChoice, float>> child_specs;
                child_specs.reserve(request.actions.size());
                for (size_t j = 0; j < request.actions.size(); ++j) {
                    child_specs.emplace_back(request.actions[j].action, priors[j]);
                }
                node.expand(child_specs, false, 0.0f);
                resolved_values[leaf_idx] = value;

                if (cfg_.use_transposition_table) {
                    transposition_table_[leaf.env.transposition_key()] = TranspositionEntry{
                        value,
                        false,
                        0.0f,
                        child_specs,
                        request.actions
                    };
                }
            }
        }

        for (size_t i = 0; i < pending.size(); ++i) {
            accumulate_value_only(pending[i].path, resolved_values[i]);
            if (profile_ != nullptr) {
                profile_->mcts_simulation.add(seconds_since(pending[i].sim_start));
            }
        }
        if (profile_ != nullptr) {
            profile_->expand_node.add(seconds_since(batch_expand_start));
        }
    }

    [[nodiscard]] static int find_board_index(
        const std::vector<BoardKey>& board_keys,
        const Semimove& sm,
        int player
    ) {
        const int target_l = sm.from.l();
        const int target_t = sm.from.t();
        const bool target_c = player == 1;
        for (size_t i = 0; i < board_keys.size(); ++i) {
            if (board_keys[i].l == target_l && board_keys[i].t == target_t && board_keys[i].c == target_c) {
                return static_cast<int>(i);
            }
        }
        for (size_t i = 0; i < board_keys.size(); ++i) {
            if (board_keys[i].l == target_l && board_keys[i].t == target_t) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    [[nodiscard]] std::vector<ActionEntry> build_action_entries(
        const CaptureKingEnv& env,
        const EncodedState& encoded,
        const std::vector<Semimove>& legal_semimoves,
        bool can_submit
    ) const {
        std::vector<ActionEntry> entries;
        entries.reserve(legal_semimoves.size() + (can_submit ? 1 : 0));
        const int player = env.current_player();
        for (const auto& sm : legal_semimoves) {
            ActionEntry entry;
            entry.action = ActionChoice{false, sm};
            entry.board_idx = find_board_index(encoded.board_keys, sm, player);
            entry.from_sq = static_cast<int64_t>(sm.from.x() + sm.from.y() * cfg_.board_side);
            entry.to_sq = static_cast<int64_t>(sm.to.x() + sm.to.y() * cfg_.board_side);
            entry.delta_t = static_cast<float>(sm.to.t() - sm.from.t());
            entry.delta_l = static_cast<float>(sm.to.l() - sm.from.l());
            entry.is_submit = 0;
            entries.push_back(entry);
        }
        if (can_submit) {
            ActionEntry entry;
            entry.action = ActionChoice{true, {}};
            entry.board_idx = -1;
            entry.is_submit = 1;
            entries.push_back(entry);
        }
        return entries;
    }

    [[nodiscard]] float expand_node(
        MCTSNode& node,
        CaptureKingEnv& env,
        float urgency,
        std::vector<ActionEntry>* action_entries_out
    ) {
        const auto expand_start = std::chrono::steady_clock::now();
        std::string tt_key;
        if (cfg_.use_transposition_table) {
            const auto tt_lookup_start = std::chrono::steady_clock::now();
            tt_key = env.transposition_key();
            auto found = transposition_table_.find(tt_key);
            if (profile_ != nullptr) {
                profile_->tt_lookup.add(seconds_since(tt_lookup_start));
            }
            if (found != transposition_table_.end()) {
                node.expand(
                    found->second.child_specs,
                    found->second.terminal,
                    found->second.terminal_value
                );
                if (action_entries_out != nullptr) {
                    *action_entries_out = found->second.action_entries;
                }
                if (profile_ != nullptr) {
                    profile_->tt_hits += 1;
                    profile_->expand_node.add(seconds_since(expand_start));
                }
                return found->second.value;
            }
            if (profile_ != nullptr) {
                profile_->tt_misses += 1;
            }
        }
        if (env.done()) {
            const float terminal_value = white_to_current_player_value(env.outcome(), env.current_player());
            node.expand({}, true, terminal_value);
            if (action_entries_out != nullptr) {
                action_entries_out->clear();
            }
            if (cfg_.use_transposition_table) {
                transposition_table_[tt_key] = TranspositionEntry{
                    terminal_value,
                    true,
                    terminal_value,
                    {},
                    {}
                };
            }
            if (profile_ != nullptr) {
                profile_->expand_node.add(seconds_since(expand_start));
            }
            return terminal_value;
        }

        const auto frontier_start = std::chrono::steady_clock::now();
        const auto [legal_semimoves, can_submit] = env.get_legal_frontier();
        if (profile_ != nullptr) {
            profile_->legal_frontier.add(seconds_since(frontier_start));
            profile_->frontier_nodes += 1;
        }
        if (legal_semimoves.empty() && !can_submit) {
            const float terminal_value = no_legal_action_terminal_value();
            node.expand({}, true, terminal_value);
            if (action_entries_out != nullptr) {
                action_entries_out->clear();
            }
            if (cfg_.use_transposition_table) {
                transposition_table_[tt_key] = TranspositionEntry{
                    terminal_value,
                    true,
                    terminal_value,
                    {},
                    {}
                };
            }
            if (profile_ != nullptr) {
                profile_->expand_node.add(seconds_since(expand_start));
            }
            return terminal_value;
        }

        const auto encode_start = std::chrono::steady_clock::now();
        const EncodedState encoded = env.encode_state();
        if (profile_ != nullptr) {
            profile_->encode_state.add(seconds_since(encode_start));
        }
        const auto build_start = std::chrono::steady_clock::now();
        std::vector<ActionEntry> entries = build_action_entries(env, encoded, legal_semimoves, can_submit);
        if (profile_ != nullptr) {
            profile_->build_action_entries.add(seconds_since(build_start));
        }
        const auto [value, logits] = network_.predict_actions(encoded, entries, urgency);
        const auto softmax_start = std::chrono::steady_clock::now();
        const std::vector<float> priors = softmax(logits);
        if (profile_ != nullptr) {
            profile_->softmax.add(seconds_since(softmax_start));
        }

        std::vector<std::pair<ActionChoice, float>> child_specs;
        child_specs.reserve(entries.size());
        for (size_t i = 0; i < entries.size(); ++i) {
            child_specs.emplace_back(entries[i].action, priors[i]);
        }
        node.expand(child_specs, false, 0.0f);

        if (action_entries_out != nullptr) {
            *action_entries_out = std::move(entries);
        }
        if (cfg_.use_transposition_table) {
            const std::vector<ActionEntry>& stored_entries =
                action_entries_out != nullptr ? *action_entries_out : entries;
            transposition_table_[tt_key] = TranspositionEntry{
                value,
                false,
                0.0f,
                child_specs,
                stored_entries
            };
        }
        if (profile_ != nullptr) {
            profile_->expand_node.add(seconds_since(expand_start));
        }
        return value;
    }

    [[nodiscard]] static std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> probs(logits.size(), 0.0f);
        if (logits.empty()) {
            return probs;
        }
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            const float clipped = std::clamp(logits[i], -20.0f, 20.0f);
            probs[i] = std::exp(clipped - max_logit);
            sum += probs[i];
        }
        if (sum <= 0.0f) {
            const float uniform = 1.0f / static_cast<float>(logits.size());
            std::fill(probs.begin(), probs.end(), uniform);
            return probs;
        }
        for (float& value : probs) {
            value /= sum;
        }
        return probs;
    }

    [[nodiscard]] std::vector<float> visit_policy(const std::vector<float>& visits, float temperature) const {
        std::vector<float> probs(visits.size(), 0.0f);
        if (visits.empty()) {
            return probs;
        }
        if (temperature < 1e-3f) {
            const size_t best = static_cast<size_t>(std::distance(visits.begin(), std::max_element(visits.begin(), visits.end())));
            probs[best] = 1.0f;
            return probs;
        }
        float sum = 0.0f;
        for (size_t i = 0; i < visits.size(); ++i) {
            probs[i] = std::pow(visits[i], 1.0f / temperature);
            sum += probs[i];
        }
        if (sum <= 0.0f) {
            const float uniform = 1.0f / static_cast<float>(visits.size());
            std::fill(probs.begin(), probs.end(), uniform);
            return probs;
        }
        for (float& value : probs) {
            value /= sum;
        }
        return probs;
    }

    [[nodiscard]] size_t sample_policy(const std::vector<float>& policy) {
        std::discrete_distribution<size_t> distribution(policy.begin(), policy.end());
        return distribution(rng_);
    }

    const SearchConfig& cfg_;
    OnnxPolicyValue& network_;
    std::mt19937& rng_;
    RunnerProfile* profile_ = nullptr;
    std::unordered_map<std::string, TranspositionEntry> transposition_table_{};
};

class SelfPlayRunner {
public:
    static constexpr float kUrgencyAlpha = 0.2f;

    SelfPlayRunner(const SearchConfig& cfg, const std::filesystem::path& model_path)
        : cfg_(cfg),
          rng_(cfg.seed),
          network_(model_path, cfg.provider, cfg.cuda_device_id, cfg.ort_intra_threads, cfg.piece_channels, cfg.board_squares, cfg.leaf_batch_size, &profile_),
          mcts_(cfg, network_, rng_, &profile_) {}

    void run() {
        run_batch(cfg_.num_games, cfg_.seed, cfg_.output_data_path, cfg_.print_games);
    }

    void serve() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") {
                std::cout << "BYE\n" << std::flush;
                return;
            }
            if (line.rfind("RUN\t", 0) != 0) {
                std::cout << "ERR\tUnknown command\n" << std::flush;
                continue;
            }

            try {
                const size_t first_tab = line.find('\t', 4);
                const size_t second_tab = line.find('\t', first_tab == std::string::npos ? first_tab : first_tab + 1);
                if (first_tab == std::string::npos || second_tab == std::string::npos) {
                    throw std::runtime_error("Malformed RUN command.");
                }
                const int num_games = std::stoi(line.substr(4, first_tab - 4));
                const uint32_t seed = static_cast<uint32_t>(std::stoul(line.substr(first_tab + 1, second_tab - first_tab - 1)));
                const std::filesystem::path output_path = line.substr(second_tab + 1);
                run_batch(num_games, seed, output_path, false);
                std::cout << "OK\t" << num_games << "\n" << std::flush;
            } catch (const std::exception& ex) {
                std::cout << "ERR\t" << ex.what() << "\n" << std::flush;
            }
        }
    }

private:
    void run_batch(int num_games, uint32_t seed, const std::filesystem::path& output_path, bool print_games) {
        const auto batch_start = std::chrono::steady_clock::now();
        rng_.seed(seed);
        std::uniform_int_distribution<int> board_limit_dist(cfg_.min_board_limit, cfg_.max_board_limit);
        std::optional<BinarySampleWriter> writer;
        if (!output_path.empty()) {
            writer.emplace(output_path);
            writer->write_header(static_cast<uint32_t>(num_games));
        }
        for (int game_idx = 1; game_idx <= num_games; ++game_idx) {
            const int board_limit = board_limit_dist(rng_);
            GameResult result = play_game(board_limit);
            if (print_games) {
                print_game(game_idx, result);
            }
            if (writer.has_value()) {
                const auto write_start = std::chrono::steady_clock::now();
                writer->write_game(result);
                profile_.binary_write_game.add(seconds_since(write_start));
            }
        }
        profile_.run_batch.add(seconds_since(batch_start));
        if (!cfg_.profile_json_path.empty()) {
            write_profile_json(cfg_.profile_json_path, profile_, cfg_);
        }
    }

    [[nodiscard]] GameResult play_game(int board_limit) {
        const auto game_start = std::chrono::steady_clock::now();
        CaptureKingEnv env(
            cfg_.variant_pgn,
            board_limit,
            cfg_.board_side,
            cfg_.board_squares,
            cfg_.line_shift,
            cfg_.material_scale
        );
        GameResult result;
        result.board_limit = board_limit;
        const bool use_max_game_length = cfg_.max_game_length > 0;
        bool terminated_by_no_action = false;

        while (!env.done() && (!use_max_game_length || env.total_semimoves() < cfg_.max_game_length)) {
            const int boards_remaining = std::max(0, board_limit - env.board_count());
            const float urgency = std::exp(-kUrgencyAlpha * static_cast<float>(boards_remaining));
            const float temperature = env.total_semimoves() < cfg_.temperature_threshold
                ? cfg_.temperature
                : cfg_.temperature_final;
            const auto encode_start = std::chrono::steady_clock::now();
            const EncodedState encoded = env.encode_state();
            profile_.encode_state.add(seconds_since(encode_start));
            const auto search_result = mcts_.select_action(env, urgency, temperature);
            if (search_result.policy_probs.empty()) {
                terminated_by_no_action = true;
                break;
            }

            const int player = env.current_player();
            result.samples.push_back(GameResult::TrainingSample{
                encoded,
                search_result.action_entries,
                search_result.policy_probs,
                urgency,
                0.0f,
                player,
            });
            if (search_result.action.is_submit) {
                const auto submit_start = std::chrono::steady_clock::now();
                const auto outcome = env.submit_turn(true);
                profile_.selfplay_submit_turn.add(seconds_since(submit_start));
                result.move_history.push_back(MoveLogEntry{
                    player,
                    true,
                    "SUBMIT",
                    search_result.root_value,
                    env.board_count(),
                });
                if (outcome.has_value()) {
                    result.outcome = *outcome;
                    result.terminal_reason = env.terminal_reason();
                }
            } else {
                const auto format_start = std::chrono::steady_clock::now();
                const std::string move_text = env.format_move(search_result.action.semimove);
                profile_.format_move.add(seconds_since(format_start));
                const auto apply_start = std::chrono::steady_clock::now();
                if (!env.apply_semimove(search_result.action.semimove, true)) {
                    throw std::runtime_error("Selected semimove failed in self-play.");
                }
                profile_.selfplay_apply_semimove.add(seconds_since(apply_start));
                result.move_history.push_back(MoveLogEntry{
                    player,
                    false,
                    move_text,
                    search_result.root_value,
                    env.board_count(),
                });
                if (env.done()) {
                    result.outcome = env.outcome();
                    result.terminal_reason = env.terminal_reason();
                }
            }
        }

        if (terminated_by_no_action && result.terminal_reason.empty()) {
            result.outcome = env.current_player() == 0 ? -1.0f : 1.0f;
            result.terminal_reason = "no_legal_action";
        } else if (!env.done() && use_max_game_length && result.terminal_reason.empty()) {
            result.outcome = 0.0f;
            result.terminal_reason = "max_game_length";
        } else if (result.terminal_reason.empty()) {
            result.outcome = env.outcome();
            result.terminal_reason = env.terminal_reason();
        }
        result.total_semimoves = env.total_semimoves();
        const auto pgn_start = std::chrono::steady_clock::now();
        result.pgn = env.show_pgn(state::SHOW_CAPTURE | state::SHOW_PROMOTION);
        profile_.show_pgn.add(seconds_since(pgn_start));
        for (auto& sample : result.samples) {
            sample.value_target = sample.player == 0 ? result.outcome : -result.outcome;
        }
        profile_.games += 1;
        profile_.semimoves += static_cast<uint64_t>(result.total_semimoves);
        profile_.samples += static_cast<uint64_t>(result.samples.size());
        profile_.play_game.add(seconds_since(game_start));
        return result;
    }

    static void print_game(int game_idx, const GameResult& game) {
        std::cout << "============================================================\n";
        std::cout << "Game #" << game_idx << "\n";
        std::cout << "============================================================\n";
        std::cout << "Outcome:     " << outcome_text(game.outcome) << "\n";
        std::cout << "Semimoves:   " << game.total_semimoves << "\n";
        std::cout << "Board limit: " << game.board_limit << "\n";
        std::cout << "Termination: " << game.terminal_reason << "\n\n";
        std::cout << "Move History:\n";
        std::cout << "--------------------------------------------------\n";
        int turn_num = 1;
        for (const auto& entry : game.move_history) {
            const char* player_text = entry.player == 0 ? "White" : "Black";
            std::cout << std::setw(5) << turn_num << ". [" << player_text << "] "
                      << std::left << std::setw(24) << entry.move_text << std::right
                      << " | boards=" << std::setw(2) << entry.board_count
                      << " | V=" << std::showpos << std::fixed << std::setprecision(4)
                      << entry.root_value << std::noshowpos << "\n";
            if (entry.player == 1 && entry.is_submit) {
                turn_num += 1;
            }
        }
        std::cout << "\n";
    }

    SearchConfig cfg_;
    RunnerProfile profile_{};
    std::mt19937 rng_;
    OnnxPolicyValue network_;
    SemimoveMcts mcts_;
};

[[nodiscard]] SearchConfig parse_args(int argc, const char* argv[], std::filesystem::path& model_path) {
    SearchConfig cfg;
    apply_variant_preset(cfg, cfg.variant_name);
    model_path = "alphazero/checkpoints/latest_fp32.onnx";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--model") {
            model_path = require_value("--model");
        } else if (arg == "--variant") {
            apply_variant_preset(cfg, require_value("--variant"));
        } else if (arg == "--games") {
            cfg.num_games = std::stoi(require_value("--games"));
        } else if (arg == "--sims") {
            cfg.num_simulations = std::stoi(require_value("--sims"));
        } else if (arg == "--leaf-batch-size") {
            cfg.leaf_batch_size = std::stoi(require_value("--leaf-batch-size"));
        } else if (arg == "--min-board-limit") {
            cfg.min_board_limit = std::stoi(require_value("--min-board-limit"));
        } else if (arg == "--max-board-limit") {
            cfg.max_board_limit = std::stoi(require_value("--max-board-limit"));
        } else if (arg == "--material-scale") {
            cfg.material_scale = std::stof(require_value("--material-scale"));
        } else if (arg == "--max-game-length") {
            cfg.max_game_length = std::stoi(require_value("--max-game-length"));
        } else if (arg == "--temperature") {
            cfg.temperature = std::stof(require_value("--temperature"));
        } else if (arg == "--temperature-final") {
            cfg.temperature_final = std::stof(require_value("--temperature-final"));
        } else if (arg == "--temperature-threshold") {
            cfg.temperature_threshold = std::stoi(require_value("--temperature-threshold"));
        } else if (arg == "--c-puct") {
            cfg.c_puct = std::stof(require_value("--c-puct"));
        } else if (arg == "--dirichlet-alpha") {
            cfg.dirichlet_alpha = std::stof(require_value("--dirichlet-alpha"));
        } else if (arg == "--dirichlet-epsilon") {
            cfg.dirichlet_epsilon = std::stof(require_value("--dirichlet-epsilon"));
        } else if (arg == "--ort-threads") {
            cfg.ort_intra_threads = std::stoi(require_value("--ort-threads"));
        } else if (arg == "--provider") {
            cfg.provider = require_value("--provider");
        } else if (arg == "--cuda-device-id") {
            cfg.cuda_device_id = std::stoi(require_value("--cuda-device-id"));
        } else if (arg == "--onnx-max-boards") {
            (void)require_value("--onnx-max-boards");
        } else if (arg == "--output-data") {
            cfg.output_data_path = require_value("--output-data");
        } else if (arg == "--profile-json") {
            cfg.profile_json_path = require_value("--profile-json");
        } else if (arg == "--quiet") {
            cfg.print_games = false;
        } else if (arg == "--seed") {
            cfg.seed = static_cast<uint32_t>(std::stoul(require_value("--seed")));
        } else if (arg == "--serve") {
            cfg.serve_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: az_selfplay_onnx [options]\n"
                << "  --model PATH                 ONNX model path (default: alphazero/checkpoints/latest_fp32.onnx)\n"
                << "  --variant NAME               Variant preset: very_small, standard, or standard_turn_zero\n"
                << "  --games N                    Number of self-play games (default: 1)\n"
                << "  --sims N                     MCTS simulations per semimove (default: 200)\n"
                << "  --leaf-batch-size N          Batched leaf evaluations per search wave (default: 1)\n"
                << "  --min-board-limit N          Minimum board limit (default: 15)\n"
                << "  --max-board-limit N          Maximum board limit (default: 25)\n"
                << "  --material-scale X           Tanh scale for board-limit material scoring (default: 2.0)\n"
                << "  --max-game-length N          Max semimoves before forced stop; <=0 disables (default: 0)\n"
                << "  --temperature X              Early-game visit temperature (default: 1.0)\n"
                << "  --temperature-final X        Late-game visit temperature (default: 0.1)\n"
                << "  --temperature-threshold N    Switch threshold in semimoves (default: 30)\n"
                << "  --c-puct X                   PUCT exploration constant (default: 2.0)\n"
                << "  --dirichlet-alpha X          Root Dirichlet alpha (default: 0.3)\n"
                << "  --dirichlet-epsilon X        Root Dirichlet mix weight (default: 0.25)\n"
                << "  --provider NAME              Execution provider: cpu or cuda (default: cpu)\n"
                << "  --cuda-device-id N           CUDA device id when provider=cuda (default: 0)\n"
                << "  --ort-threads N              ORT intra-op threads (default: 1)\n"
                << "  --output-data PATH           Write binary training samples to PATH\n"
                << "  --profile-json PATH          Write detailed C++ timing profile JSON to PATH\n"
                << "  --serve                      Run as a persistent stdin/stdout worker\n"
                << "  --quiet                      Suppress human-readable game logs\n"
                << "  --seed N                     RNG seed (default: 1)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.min_board_limit > cfg.max_board_limit) {
        throw std::runtime_error("min-board-limit must be <= max-board-limit.");
    }
    if (cfg.provider != "cpu" && cfg.provider != "cuda") {
        throw std::runtime_error("provider must be 'cpu' or 'cuda'.");
    }
    if (cfg.leaf_batch_size <= 0) {
        throw std::runtime_error("leaf-batch-size must be >= 1.");
    }
    if (cfg.cuda_device_id < 0) {
        throw std::runtime_error("cuda-device-id must be >= 0.");
    }
    return cfg;
}

}  // namespace

int main(int argc, const char* argv[]) {
    try {
        std::filesystem::path model_path;
        SearchConfig cfg = parse_args(argc, argv, model_path);
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("Model path does not exist: " + model_path.string());
        }
        SelfPlayRunner runner(cfg, model_path);
        if (cfg.serve_mode) {
            runner.serve();
        } else {
            runner.run();
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
