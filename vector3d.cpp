#include "vector3d.h"

vector3d util::getMaxCornerDim(vec3 centre, vector3d aCorner, vector3d aCentre, vector3d bCorner, vector3d bCentre) {

	float x = std::max(std::abs(aCorner.pos[0] - centre[0]) + std::abs(aCentre.pos[0] - centre[0]), std::abs(bCorner.pos[0] - centre[0]) + std::abs(bCentre.pos[0] - centre[0]));
	float y = std::max(std::abs(aCorner.pos[1] - centre[1]) + std::abs(aCentre.pos[1] - centre[1]), std::abs(bCorner.pos[1] - centre[1]) + std::abs(bCentre.pos[1] - centre[1]));
	float z = std::max(std::abs(aCorner.pos[2] - centre[2]) + std::abs(aCentre.pos[2] - centre[2]), std::abs(bCorner.pos[2] - centre[2]) + std::abs(bCentre.pos[2] - centre[2]));

	vector3d b = vector3d();
	b.pos[0] = x;
	b.pos[1] = y;
	b.pos[2] = z;
	return b;

}