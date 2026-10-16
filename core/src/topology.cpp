#include "catan/topology.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <compare>
#include <utility>
#include <vector>

using namespace std;

namespace catan {
namespace {

// Exact board position: x in units of sqrt(3)/2, y in units of 1/2 (see topology.h).
// Members are ordered (y, x) so the defaulted comparison sorts top-to-bottom, left-to-right.
struct Point {
    int y, x;
    auto operator<=>(const Point&) const = default;
};

// Corner offsets from a hex centre, clockwise from the top corner.
constexpr array<Point, 6> CORNER_OFFSETS = {{
    {-2, 0},   // top
    {-1, 1},   // upper-right
    {1, 1},    // lower-right
    {2, 0},    // bottom
    {1, -1},   // lower-left
    {-1, -1},  // upper-left
}};

enum Side : uint8_t { NE, E, SE, SW, W, NW };

struct PortLocation {
    uint8_t hex;
    Side side;
};

// Base-game harbor positions, clockwise from the top-left harbor.
constexpr array<PortLocation, NUM_PORTS> PORT_LOCATIONS = {{
    {0, NW},
    {1, NE},
    {6, NE},
    {11, E},
    {15, SE},
    {17, SE},
    {16, SW},
    {12, W},
    {3, W},
}};

Point hex_center(HexCoord c) {
    return {3 * c.r, 2 * c.q + c.r};
}

Point corner(HexCoord c, int i) {
    const Point center = hex_center(c);
    return {center.y + CORNER_OFFSETS[i].y, center.x + CORNER_OFFSETS[i].x};
}

// Twice the midpoint of side i, which keeps it on the integer grid.
Point doubled_side_midpoint(HexCoord c, int i) {
    const Point a = corner(c, i);
    const Point b = corner(c, (i + 1) % 6);
    return {a.y + b.y, a.x + b.x};
}

// Sorted, deduplicated positions; a position's index is its canonical id.
vector<Point> canonical(vector<Point> points) {
    sort(points.begin(), points.end());
    points.erase(unique(points.begin(), points.end()), points.end());
    return points;
}

uint8_t id_of(const vector<Point>& sorted, Point p) {
    const auto it = lower_bound(sorted.begin(), sorted.end(), p);
    assert(it != sorted.end() && *it == p);
    return static_cast<uint8_t>(it - sorted.begin());
}

// Writes value into the first NONE slot of a NONE-padded list.
template <size_t N>
void append(array<uint8_t, N>& list, uint8_t value) {
    const auto slot = find(list.begin(), list.end(), NONE);
    assert(slot != list.end());
    *slot = value;
}

// NONE (0xFF) sorts last, so sorting the whole array keeps the padding at the end.
template <size_t N>
void sort_padded(array<uint8_t, N>& list) {
    sort(list.begin(), list.end());
}

template <size_t N>
void fill_none(array<array<uint8_t, N>, NUM_VERTICES>& lists) {
    for (auto& list : lists) list.fill(NONE);
}

[[maybe_unused]] int edge_set_count(const EdgeSet& es) {
    return popcount(es[0]) + popcount(es[1]);
}

Topology build_topology() {
    Topology t{};

    // Hexes: row-major over a radius-2 axial hexagon.
    int h = 0;
    for (int r = -2; r <= 2; ++r) {
        const int q_min = max(-2, -2 - r);
        const int q_max = min(2, 2 - r);
        for (int q = q_min; q <= q_max; ++q) {
            t.hex_coord[h++] = {static_cast<int8_t>(q), static_cast<int8_t>(r)};
        }
    }
    assert(h == NUM_HEXES);

    // Canonical vertex and edge ids from exact positions; shared corners and sides collapse.
    vector<Point> corner_points, side_points;
    for (const HexCoord c : t.hex_coord) {
        for (int i = 0; i < 6; ++i) {
            corner_points.push_back(corner(c, i));
            side_points.push_back(doubled_side_midpoint(c, i));
        }
    }
    const vector<Point> vertex_pos = canonical(std::move(corner_points));
    const vector<Point> edge_pos = canonical(std::move(side_points));
    assert(static_cast<int>(vertex_pos.size()) == NUM_VERTICES);
    assert(static_cast<int>(edge_pos.size()) == NUM_EDGES);

    // Hex -> vertex/edge incidence, and edge -> vertex endpoints.
    array<array<uint8_t, 2>, NUM_EDGES> edge_hexes;
    for (auto& list : edge_hexes) list.fill(NONE);
    for (int hex = 0; hex < NUM_HEXES; ++hex) {
        const HexCoord c = t.hex_coord[hex];
        for (int i = 0; i < 6; ++i) {
            t.hex_vertices[hex][i] = id_of(vertex_pos, corner(c, i));
            t.hex_edges[hex][i] = id_of(edge_pos, doubled_side_midpoint(c, i));
        }
        for (int i = 0; i < 6; ++i) {
            const uint8_t e = t.hex_edges[hex][i];
            const uint8_t a = t.hex_vertices[hex][i];
            const uint8_t b = t.hex_vertices[hex][(i + 1) % 6];
            t.edge_vertices[e] = {min(a, b), max(a, b)};
            append(edge_hexes[e], static_cast<uint8_t>(hex));
        }
    }

    // Hex neighbors: the other hex sharing each side, if any.
    for (int hex = 0; hex < NUM_HEXES; ++hex) {
        for (int i = 0; i < 6; ++i) {
            const auto& [a, b] = edge_hexes[t.hex_edges[hex][i]];
            t.hex_neighbors[hex][i] = (a == hex) ? b : a;
        }
    }

    // Vertex incidence lists. Hexes and edges are visited in ascending order, so those lists
    // come out sorted; vertex neighbors are sorted explicitly.
    fill_none(t.vertex_hexes);
    fill_none(t.vertex_vertices);
    fill_none(t.vertex_edges);
    for (int hex = 0; hex < NUM_HEXES; ++hex) {
        for (const uint8_t v : t.hex_vertices[hex]) append(t.vertex_hexes[v], static_cast<uint8_t>(hex));
    }
    for (int e = 0; e < NUM_EDGES; ++e) {
        const auto [a, b] = t.edge_vertices[e];
        append(t.vertex_edges[a], static_cast<uint8_t>(e));
        append(t.vertex_edges[b], static_cast<uint8_t>(e));
        append(t.vertex_vertices[a], b);
        append(t.vertex_vertices[b], a);
    }
    for (auto& list : t.vertex_vertices) sort_padded(list);

    // Edge adjacency: every other edge at either endpoint (at most 2 + 2).
    for (int e = 0; e < NUM_EDGES; ++e) {
        t.edge_edges[e].fill(NONE);
        for (const uint8_t v : t.edge_vertices[e]) {
            for (const uint8_t other : t.vertex_edges[v]) {
                if (other != NONE && other != e) append(t.edge_edges[e], other);
            }
        }
        sort_padded(t.edge_edges[e]);
    }

    // Bitset views, derived from the lists above.
    for (int v = 0; v < NUM_VERTICES; ++v) {
        for (const uint8_t n : t.vertex_vertices[v]) {
            if (n != NONE) t.vertex_neighbor_mask[v] |= vertex_bit(n);
        }
        for (const uint8_t e : t.vertex_edges[v]) {
            if (e != NONE) edge_set_add(t.vertex_edge_mask[v], e);
        }
    }
    for (int e = 0; e < NUM_EDGES; ++e) {
        for (const uint8_t n : t.edge_edges[e]) {
            if (n != NONE) edge_set_add(t.edge_adjacent_mask[e], n);
        }
    }

    // Coast: edges bordering one land hex, and their endpoints.
    for (int e = 0; e < NUM_EDGES; ++e) {
        if (edge_hexes[e][1] != NONE) continue;
        edge_set_add(t.coastal_edges, e);
        for (const uint8_t v : t.edge_vertices[e]) t.coastal_vertices |= vertex_bit(v);
    }
    assert(edge_set_count(t.coastal_edges) == 30);
    assert(popcount(t.coastal_vertices) == 30);

    // Harbor slots.
    for (int p = 0; p < NUM_PORTS; ++p) {
        const auto [hex, side] = PORT_LOCATIONS[p];
        const uint8_t e = t.hex_edges[hex][side];
        assert(t.hex_neighbors[hex][side] == NONE);
        assert(edge_set_has(t.coastal_edges, e));
        t.port_slots[p] = {e, t.edge_vertices[e][0], t.edge_vertices[e][1], hex};
    }

    return t;
}

}  // namespace

const Topology& topology() {
    static const Topology t = build_topology();
    return t;
}

}  // namespace catan
