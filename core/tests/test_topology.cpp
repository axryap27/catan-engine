#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "catan/topology.h"
#include "catan/types.h"

using namespace std;
using namespace catan;

namespace {

const Topology& T() { return topology(); }

// Non-NONE entries of a padded list, as a set.
template <size_t N>
set<int> as_set(const array<uint8_t, N>& list) {
    set<int> s;
    for (uint8_t x : list) if (x != NONE) s.insert(x);
    return s;
}

template <size_t N>
int real_count(const array<uint8_t, N>& list) {
    int n = 0;
    for (uint8_t x : list) if (x != NONE) ++n;
    return n;
}

// Padding is at the end and real entries are strictly ascending.
template <size_t N>
bool padded_ascending(const array<uint8_t, N>& list) {
    for (size_t i = 1; i < N; ++i) {
        if (list[i - 1] == NONE && list[i] != NONE) return false;
        if (list[i] != NONE && list[i - 1] >= list[i]) return false;
    }
    return true;
}

template <size_t N>
bool has(const array<uint8_t, N>& list, int x) {
    return find(list.begin(), list.end(), static_cast<uint8_t>(x)) != list.end();
}

bool vset_has(VertexSet s, int v) { return (s >> v) & 1; }

set<int> edge_set_members(const EdgeSet& es) {
    set<int> s;
    for (int e = 0; e < NUM_EDGES; ++e) if (edge_set_has(es, e)) s.insert(e);
    return s;
}
set<int> vertex_set_members(VertexSet vs) {
    set<int> s;
    for (int v = 0; v < NUM_VERTICES; ++v) if (vset_has(vs, v)) s.insert(v);
    return s;
}

// Number of land hexes each edge borders, from hex_edges.
array<int, NUM_EDGES> edge_hex_count() {
    array<int, NUM_EDGES> c{};
    for (int h = 0; h < NUM_HEXES; ++h)
        for (int i = 0; i < 6; ++i) ++c[T().hex_edges[h][i]];
    return c;
}

// Exact positions recomputed independently: (y, x), x in sqrt(3)/2 units, y in 1/2 units.
using Pt = pair<int, int>;
constexpr int DY[6] = {-2, -1, 1, 2, 1, -1};
constexpr int DX[6] = {0, 1, 1, 0, -1, -1};
Pt corner_pos(int h, int i) {
    const HexCoord c = T().hex_coord[h];
    return {3 * c.r + DY[i], 2 * c.q + c.r + DX[i]};
}

}  // namespace

TEST_CASE("topology: returns the same object on repeated calls", "[topology]") {
    REQUIRE(&topology() == &topology());
}

TEST_CASE("topology: sizes and hex geometry", "[topology]") {
    const Topology& t = T();
    CHECK(t.hex_coord.size() == 19);
    CHECK(t.vertex_hexes.size() == 54);
    CHECK(t.edge_vertices.size() == 72);
    CHECK(t.port_slots.size() == 9);

    CHECK(t.hex_coord[9].q == 0);
    CHECK(t.hex_coord[9].r == 0);

    // Row-major order and the documented rows.
    const int row_len[5] = {3, 4, 5, 4, 3};
    int h = 0;
    for (int ri = 0; ri < 5; ++ri) {
        const int r = ri - 2;
        const int q_min = max(-2, -2 - r);
        for (int k = 0; k < row_len[ri]; ++k, ++h) {
            CHECK(t.hex_coord[h].r == r);
            CHECK(t.hex_coord[h].q == q_min + k);
        }
    }
    CHECK(h == 19);
}

TEST_CASE("topology: each hex has 6 distinct vertices and edges, edge i joins corners i and i+1",
          "[topology]") {
    const Topology& t = T();
    for (int h = 0; h < NUM_HEXES; ++h) {
        CAPTURE(h);
        set<int> vs, es;
        for (int i = 0; i < 6; ++i) {
            CHECK(t.hex_vertices[h][i] < NUM_VERTICES);
            CHECK(t.hex_edges[h][i] < NUM_EDGES);
            vs.insert(t.hex_vertices[h][i]);
            es.insert(t.hex_edges[h][i]);
        }
        CHECK(vs.size() == 6);
        CHECK(es.size() == 6);
        for (int i = 0; i < 6; ++i) {
            CAPTURE(i);
            const int a = t.hex_vertices[h][i];
            const int b = t.hex_vertices[h][(i + 1) % 6];
            const auto& ev = t.edge_vertices[t.hex_edges[h][i]];
            CHECK(ev[0] == min(a, b));
            CHECK(ev[1] == max(a, b));
        }
    }
}

TEST_CASE("topology: vertex ids are positional and grouped in documented rows", "[topology]") {
    const Topology& t = T();
    // Tops of hexes 0..2 are vertices 0..2.
    for (int h = 0; h < 3; ++h) CHECK(t.hex_vertices[h][0] == h);

    // Recompute each vertex's position from every hex that touches it; they must agree.
    map<int, Pt> pos;
    for (int h = 0; h < NUM_HEXES; ++h) {
        for (int i = 0; i < 6; ++i) {
            const int v = t.hex_vertices[h][i];
            const Pt p = corner_pos(h, i);
            auto [it, inserted] = pos.emplace(v, p);
            if (!inserted) CHECK(it->second == p);
        }
    }
    REQUIRE(pos.size() == 54);
    // Ids strictly increase in (y, x).
    for (int v = 1; v < NUM_VERTICES; ++v) CHECK(pos[v - 1] < pos[v]);
    // Per-row counts.
    const vector<int> expect = {3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3};
    vector<int> rows;
    int last_y = -1000;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        if (pos[v].first != last_y) { rows.push_back(0); last_y = pos[v].first; }
        ++rows.back();
    }
    CHECK(rows == expect);
}

TEST_CASE("topology: edge ids are positional and grouped in documented rows", "[topology]") {
    const Topology& t = T();
    map<int, Pt> pos;  // doubled midpoints
    for (int h = 0; h < NUM_HEXES; ++h) {
        for (int i = 0; i < 6; ++i) {
            const Pt a = corner_pos(h, i), b = corner_pos(h, (i + 1) % 6);
            const Pt p{a.first + b.first, a.second + b.second};
            auto [it, inserted] = pos.emplace(int{t.hex_edges[h][i]}, p);
            if (!inserted) CHECK(it->second == p);
        }
    }
    REQUIRE(pos.size() == 72);
    for (int e = 1; e < NUM_EDGES; ++e) CHECK(pos[e - 1] < pos[e]);
    const vector<int> expect = {6, 4, 8, 5, 10, 6, 10, 5, 8, 4, 6};
    vector<int> rows;
    int last_y = -1000;
    for (int e = 0; e < NUM_EDGES; ++e) {
        if (pos[e].first != last_y) { rows.push_back(0); last_y = pos[e].first; }
        ++rows.back();
    }
    CHECK(rows == expect);
}

TEST_CASE("topology: vertex degree and incidence", "[topology]") {
    const Topology& t = T();
    int deg2 = 0, deg3 = 0;
    for (int v = 0; v < NUM_VERTICES; ++v) {
        CAPTURE(v);
        const int d = real_count(t.vertex_vertices[v]);
        CHECK((d == 2 || d == 3));
        CHECK(real_count(t.vertex_edges[v]) == d);
        (d == 2 ? deg2 : deg3)++;
        const int nh = real_count(t.vertex_hexes[v]);
        CHECK((nh >= 1 && nh <= 3));
        // vertex_hexes agrees with hex_vertices.
        for (int h = 0; h < NUM_HEXES; ++h)
            CHECK(has(t.hex_vertices[h], v) == has(t.vertex_hexes[v], h));
    }
    CHECK(deg2 + deg3 == 54);
    CHECK(deg2 == 18);
    CHECK(deg3 == 36);
}

TEST_CASE("topology: coastal vertices and edges", "[topology]") {
    const Topology& t = T();
    const auto cnt = edge_hex_count();
    set<int> exp_edges, exp_verts;
    for (int e = 0; e < NUM_EDGES; ++e) {
        CHECK((cnt[e] == 1 || cnt[e] == 2));
        if (cnt[e] == 1) exp_edges.insert(e);
    }
    for (int v = 0; v < NUM_VERTICES; ++v)
        if (real_count(t.vertex_hexes[v]) < 3) exp_verts.insert(v);

    CHECK(exp_edges.size() == 30);
    CHECK(exp_verts.size() == 30);
    CHECK(edge_set_members(t.coastal_edges) == exp_edges);
    CHECK(vertex_set_members(t.coastal_vertices) == exp_verts);
    CHECK((t.coastal_vertices >> 54) == 0);
    CHECK(((t.coastal_edges[1] >> (72 - 64)) == 0));
}

TEST_CASE("topology: adjacency lists are padded, ascending and symmetric", "[topology]") {
    const Topology& t = T();
    for (int v = 0; v < NUM_VERTICES; ++v) {
        CAPTURE(v);
        CHECK(padded_ascending(t.vertex_hexes[v]));
        CHECK(padded_ascending(t.vertex_vertices[v]));
        CHECK(padded_ascending(t.vertex_edges[v]));
        for (uint8_t n : t.vertex_vertices[v]) {
            if (n == NONE) continue;
            CHECK(n != v);
            CHECK(has(t.vertex_vertices[n], v));
        }
    }
    for (int e = 0; e < NUM_EDGES; ++e) {
        CAPTURE(e);
        CHECK(padded_ascending(t.edge_edges[e]));
        CHECK(t.edge_vertices[e][0] < t.edge_vertices[e][1]);
        const int d = real_count(t.edge_edges[e]);
        CHECK((d >= 2 && d <= 4));
        for (uint8_t n : t.edge_edges[e]) {
            if (n == NONE) continue;
            CHECK(n != e);
            CHECK(has(t.edge_edges[n], e));
        }
    }
    for (int h = 0; h < NUM_HEXES; ++h) {
        for (int i = 0; i < 6; ++i) {
            const uint8_t n = t.hex_neighbors[h][i];
            if (n == NONE) continue;
            CAPTURE(h, i);
            REQUIRE(n < NUM_HEXES);
            CHECK(n != h);
            CHECK(has(t.hex_neighbors[n], h));
            // Opposite side (i + 3) % 6 points back.
            CHECK(t.hex_neighbors[n][(i + 3) % 6] == h);
        }
    }
}

TEST_CASE("topology: edges, vertices and edge-edge agree", "[topology]") {
    const Topology& t = T();
    for (int e = 0; e < NUM_EDGES; ++e) {
        CAPTURE(e);
        const int a = t.edge_vertices[e][0], b = t.edge_vertices[e][1];
        CHECK(has(t.vertex_vertices[a], b));
        CHECK(has(t.vertex_vertices[b], a));
        CHECK(has(t.vertex_edges[a], e));
        CHECK(has(t.vertex_edges[b], e));
        // edge_edges = union of the other edges at both endpoints.
        set<int> exp;
        for (int v : {a, b})
            for (uint8_t o : t.vertex_edges[v]) if (o != NONE && o != e) exp.insert(o);
        CHECK(as_set(t.edge_edges[e]) == exp);
    }
    for (int v = 0; v < NUM_VERTICES; ++v) {
        CAPTURE(v);
        set<int> nbrs;
        for (uint8_t e : t.vertex_edges[v]) {
            if (e == NONE) continue;
            const auto& ev = t.edge_vertices[e];
            REQUIRE((ev[0] == v || ev[1] == v));
            nbrs.insert(ev[0] == v ? ev[1] : ev[0]);
        }
        CHECK(nbrs == as_set(t.vertex_vertices[v]));
    }
    // Each edge appears in exactly two vertex_edges lists.
    array<int, NUM_EDGES> seen{};
    for (int v = 0; v < NUM_VERTICES; ++v)
        for (uint8_t e : t.vertex_edges[v]) if (e != NONE) ++seen[e];
    for (int e = 0; e < NUM_EDGES; ++e) CHECK(seen[e] == 2);
}

TEST_CASE("topology: bitmasks equal their lists", "[topology]") {
    const Topology& t = T();
    for (int v = 0; v < NUM_VERTICES; ++v) {
        CAPTURE(v);
        CHECK(vertex_set_members(t.vertex_neighbor_mask[v]) == as_set(t.vertex_vertices[v]));
        CHECK(edge_set_members(t.vertex_edge_mask[v]) == as_set(t.vertex_edges[v]));
    }
    for (int e = 0; e < NUM_EDGES; ++e) {
        CAPTURE(e);
        CHECK(edge_set_members(t.edge_adjacent_mask[e]) == as_set(t.edge_edges[e]));
    }
}

TEST_CASE("topology: hex neighbours", "[topology]") {
    const Topology& t = T();
    int interior = 0, ring3 = 0, ring4 = 0, total_links = 0;
    for (int h = 0; h < NUM_HEXES; ++h) {
        CAPTURE(h);
        int n = 0;
        for (int i = 0; i < 6; ++i) {
            const uint8_t nb = t.hex_neighbors[h][i];
            if (nb == NONE) {
                // Sea side: edge is coastal.
                CHECK(edge_set_has(t.coastal_edges, t.hex_edges[h][i]));
                continue;
            }
            ++n;
            CAPTURE(i, int{nb});
            CHECK(has(t.hex_edges[nb], t.hex_edges[h][i]));
            CHECK(!edge_set_has(t.coastal_edges, t.hex_edges[h][i]));
            // Axial offsets for sides NE,E,SE,SW,W,NW.
            static constexpr int dq[6] = {1, 1, 0, -1, -1, 0};
            static constexpr int dr[6] = {-1, 0, 1, 1, 0, -1};
            CHECK(t.hex_coord[nb].q == t.hex_coord[h].q + dq[i]);
            CHECK(t.hex_coord[nb].r == t.hex_coord[h].r + dr[i]);
        }
        total_links += n;
        const int dist = max({abs(t.hex_coord[h].q), abs(t.hex_coord[h].r),
                              abs(t.hex_coord[h].q + t.hex_coord[h].r)});
        if (dist < 2) { CHECK(n == 6); ++interior; }
        else { CHECK((n == 3 || n == 4)); (n == 3 ? ring3 : ring4)++; }
    }
    CHECK(interior == 7);
    CHECK(ring3 == 6);
    CHECK(ring4 == 6);
    CHECK(total_links == 2 * 42);  // 72 edges - 30 coastal = 42 shared
}

TEST_CASE("topology: port slots match the documented table", "[topology]") {
    const Topology& t = T();
    struct Row { int edge, v0, v1, hex; };
    const array<Row, 9> table = {{
        {0, 0, 3, 0}, {3, 1, 5, 1}, {17, 10, 15, 6}, {38, 26, 32, 11}, {61, 42, 46, 15},
        {69, 49, 52, 17}, {66, 47, 51, 16}, {49, 33, 38, 12}, {18, 11, 16, 3},
    }};
    for (int p = 0; p < NUM_PORTS; ++p) {
        CAPTURE(p);
        const PortSlot& s = t.port_slots[p];
        CHECK(s.edge == table[p].edge);
        CHECK(s.v0 == table[p].v0);
        CHECK(s.v1 == table[p].v1);
        CHECK(s.hex == table[p].hex);
    }
}

TEST_CASE("topology: port slots are consistent and disjoint", "[topology]") {
    const Topology& t = T();
    set<int> edges, verts;
    for (int p = 0; p < NUM_PORTS; ++p) {
        CAPTURE(p);
        const PortSlot& s = t.port_slots[p];
        REQUIRE(s.edge < NUM_EDGES);
        CHECK(edge_set_has(t.coastal_edges, s.edge));
        CHECK(has(t.hex_edges[s.hex], s.edge));
        CHECK(t.edge_vertices[s.edge][0] == s.v0);
        CHECK(t.edge_vertices[s.edge][1] == s.v1);
        CHECK(s.v0 < s.v1);
        CHECK(edges.insert(s.edge).second);
        CHECK(verts.insert(s.v0).second);
        CHECK(verts.insert(s.v1).second);
    }
    CHECK(edges.size() == 9);
    CHECK(verts.size() == 18);
}

TEST_CASE("topology: coastal gaps between harbors run 2,3,2,2,3,2,2,3,2", "[topology]") {
    const Topology& t = T();
    // Walk the coastal cycle: coastal edges adjacent via a shared vertex form one 30-cycle.
    const int start = t.port_slots[0].edge;
    vector<int> cycle{start};
    int prev = -1, cur = start;
    while (true) {
        int next = -1;
        for (uint8_t n : t.edge_edges[cur])
            if (n != NONE && n != prev && edge_set_has(t.coastal_edges, n)) {
                // Prefer continuing in a consistent direction: pick the first on the first step.
                if (prev == -1 && cycle.size() == 1) { next = n; break; }
                if (next == -1) next = n;
            }
        REQUIRE(next != -1);
        if (next == start) break;
        cycle.push_back(next);
        prev = cur;
        cur = next;
        REQUIRE(cycle.size() <= 30);
    }
    REQUIRE(cycle.size() == 30);
    set<int> port_edges;
    for (const auto& s : t.port_slots) port_edges.insert(s.edge);
    vector<int> at;
    for (int i = 0; i < 30; ++i) if (port_edges.count(cycle[i])) at.push_back(i);
    REQUIRE(at.size() == 9);
    vector<int> gaps;
    for (size_t i = 0; i < 9; ++i) gaps.push_back((at[(i + 1) % 9] - at[i] + 30) % 30 - 1);
    // Direction of the walk is arbitrary, so accept the sequence or its reverse, rotated.
    const vector<int> g = {2, 3, 2, 2, 3, 2, 2, 3, 2};
    bool ok = false;
    for (int rev = 0; rev < 2 && !ok; ++rev) {
        vector<int> s = gaps;
        if (rev) reverse(s.begin(), s.end());
        for (size_t rot = 0; rot < 9 && !ok; ++rot) {
            rotate(s.begin(), s.begin() + 1, s.end());
            ok = (s == g);
        }
    }
    CHECK(ok);
}
