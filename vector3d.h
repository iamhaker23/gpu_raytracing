#pragma once

#include <algorithm>
#include <linmath.h>
//Wrapper for vec3 since you cannot store an array directly in an STL vector object
struct vector3d {

	vec3 pos = { 0 };

};

class util {
public:
	static vector3d getMaxCornerDim(vec3 centre, vector3d aCorner, vector3d aCentre, vector3d bCorner, vector3d bCentre);
};

