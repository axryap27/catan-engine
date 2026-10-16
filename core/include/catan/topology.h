#pragma once
#include <array>
#include <cstdint>

#include "catan/types.h"

// Immutable board topology for the 4-player base game: 19 land hexes, 54 vertices, 72 edges,
// 9 harbor slots. Built once on first use and shared; it never enters game state.
//
// Geometry
//   Pointy-top hexes in axial coordinates (q, r); r grows downward, q grows to the right.
//   Pixel positions are kept exact by measuring x in units of sqrt(3)/2 and y in units of 1/2
//   (hex size 1), so a hex centre sits at (2q + r, 3r) and its corners at integer offsets.
//
// Hexes: row-major, top to bottom, left to right within a row (rows of 3, 4, 5, 4, 3).
//          row r = -2:   0  1  2              q =  0..2
//          row r = -1:  3  4  5  6            q = -1..2
//          row r =  0: 7  8  9 10 11          q = -2..2
//          row r =  1:  12 13 14 15           q = -2..1
//          row r =  2:   16 17 18             q = -2..0
//   Hex 9 is the centre (0, 0).
//
// Hex corners and sides (index i into hex_vertices / hex_edges / hex_neighbors):
//   corners, clockwise from the top: 0 top, 1 upper-right, 2 lower-right,
//                                    3 bottom, 4 lower-left, 5 upper-left
//   side i joins corner i and corner (i + 1) % 6:
//                                    0 NE, 1 E, 2 SE, 3 SW, 4 W, 5 NW
//
// Vertices: numbered by position, sorted by (y, x): top to bottom, left to right. Vertex rows
//   have 3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3 vertices (ids 0-2, 3-6, 7-10, 11-15, 16-20,
//   21-26, 27-32, 33-37, 38-42, 43-46, 47-50, 51-53).
//
// Edges: numbered by midpoint position, sorted by (y, x) the same way. Edge rows have
//   6, 4, 8, 5, 10, 6, 10, 5, 8, 4, 6 edges (slanted and vertical rows alternate).
//
// Port slots: the 9 harbor positions of the printed base-game frame (same layout as the
//   official rulebook and Catanatron's BASE_MAP_TEMPLATE), clockwise from the top-left one.
//   Coastal-edge gaps between consecutive harbors run 2, 3, 2, 2, 3, 2, 2, 3, 2.
//     slot  hex  side   edge  vertices
//      0     0   NW       0    0,  3
//      1     1   NE       3    1,  5
//      2     6   NE      17   10, 15
//      3    11   E       38   26, 32
//      4    15   SE      61   42, 46
//      5    17   SE      69   49, 52
//      6    16   SW      66   47, 51
//      7    12   W       49   33, 38
//      8     3   W       18   11, 16
//   The (hex, side) pairs are the source of truth; edge and vertex ids follow from them.

namespace catan {

struct HexCoord {
    int8_t q, r;
};

// A harbor position: one coastal edge, its two vertices (ascending), and the land hex it borders.
struct PortSlot {
    uint8_t edge;
    uint8_t v0, v1;
    uint8_t hex;
};

struct Topology {
    std::array<HexCoord, NUM_HEXES> hex_coord;
    std::array<std::array<uint8_t, 6>, NUM_HEXES> hex_vertices;   // clockwise from the top corner
    std::array<std::array<uint8_t, 6>, NUM_HEXES> hex_edges;      // edge i joins vertices i, i+1
    std::array<std::array<uint8_t, 6>, NUM_HEXES> hex_neighbors;  // across edge i, NONE if sea
    std::array<std::array<uint8_t, 3>, NUM_VERTICES> vertex_hexes;     // NONE-padded, ascending
    std::array<std::array<uint8_t, 3>, NUM_VERTICES> vertex_vertices;  // NONE-padded, ascending
    std::array<std::array<uint8_t, 3>, NUM_VERTICES> vertex_edges;     // NONE-padded, ascending
    std::array<std::array<uint8_t, 2>, NUM_EDGES> edge_vertices;       // ascending
    std::array<std::array<uint8_t, 4>, NUM_EDGES> edge_edges;  // share a vertex; NONE-padded, asc.
    std::array<VertexSet, NUM_VERTICES> vertex_neighbor_mask;  // = vertex_vertices as a set
    std::array<EdgeSet, NUM_VERTICES> vertex_edge_mask;        // = vertex_edges as a set
    std::array<EdgeSet, NUM_EDGES> edge_adjacent_mask;         // = edge_edges as a set
    std::array<PortSlot, NUM_PORTS> port_slots;
    VertexSet coastal_vertices;  // vertices touching fewer than 3 land hexes
    EdgeSet coastal_edges;       // edges bordering exactly one land hex
};

// Built on first call (function-local static), immutable afterwards.
const Topology& topology();

}  // namespace catan
