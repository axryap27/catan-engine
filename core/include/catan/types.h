#pragma once
#include <array>
#include <cstdint>

namespace catan {

inline constexpr int NUM_PLAYERS  = 4;
inline constexpr int NUM_HEXES    = 19;
inline constexpr int NUM_VERTICES = 54;
inline constexpr int NUM_EDGES    = 72;
inline constexpr int NUM_PORTS    = 9;
inline constexpr int NUM_RES      = 5;
inline constexpr int NUM_DEV      = 5;
inline constexpr uint8_t NONE     = 0xFF;   // "no player / no hex / no index" sentinel

// Resource order is canonical: hex types, port types and hand arrays all index by it.
enum class Resource : uint8_t { BRICK, LUMBER, WOOL, GRAIN, ORE };
enum class HexType  : uint8_t { BRICK, LUMBER, WOOL, GRAIN, ORE, DESERT };   // first 5 match Resource
enum class PortType : uint8_t { BRICK, LUMBER, WOOL, GRAIN, ORE, GENERIC };  // 2:1 per resource, GENERIC is 3:1
enum class DevCard  : uint8_t { KNIGHT, VICTORY_POINT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY };

enum class Phase : uint8_t {
    SETUP, ROLL, DISCARD, MOVE_ROBBER, STEAL, MAIN,
    ROAD_BUILDING, YOP_PICK, MONOPOLY_PICK, EXCHANGE_RECV, TERMINAL
};

// 72-bit edge set: bit e lives in word e / 64, bit e % 64.
using EdgeSet = std::array<uint64_t, 2>;
// 54-bit vertex set fits in one word.
using VertexSet = uint64_t;

constexpr int idx(Resource r) { return static_cast<int>(r); }
constexpr int idx(DevCard d)  { return static_cast<int>(d); }

constexpr bool edge_set_has(const EdgeSet& es, int e) {
    return (es[e / 64] >> (e % 64)) & 1;
}

constexpr void edge_set_add(EdgeSet& es, int e) {
    es[e / 64] |= uint64_t{1} << (e % 64);
}

constexpr uint64_t vertex_bit(int v) {
    return uint64_t{1} << v;
}

}  // namespace catan
