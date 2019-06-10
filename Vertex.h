#pragma once
#include "linmath.h"

//TODO: handle chunking correctly (need to add chunked BVH at "count time" but currently uses pointer) 
#define BVH_CHUNK_SIZE 4000

#define BVH_BOX_SIZE 250.0f
#define VERT_IMPORT_SCALE 1000.0f

struct BVH {
	vec3 min = { 0 };
	vec3 max = { 0 };
	int numTris;
	int triIdx[BVH_CHUNK_SIZE] = { 0 };

	int depth = 0;
	int children[8] = { 0 };
	int nextOctree = -1;

	BVH() {
		for (int i = 0; i < BVH_CHUNK_SIZE; i++) {
			triIdx[i] = -1;
		}
		for (int i = 0; i < 8; i++) {
			children[i] = -1;
		}
	}

};

struct BVH_BAKE {
	vec3 min = { 0 };
	vec3 max = { 0 };
	std::vector<int> triIdx;

	bool hasVerts() {
		return triIdx.size() > 0;
	}

	BVH_BAKE() {
		triIdx = std::vector<int>();
		children = std::vector<BVH_BAKE>();

	}

	std::vector<BVH_BAKE> children;

	void refreshChildren() {
		
		if (children.size() < 8) {
			for (int c = 0; c < 8; c++) {
				children.push_back(BVH_BAKE());
			}
		}

		vec3 xStep = { 0 };
		vec3 yStep = { 0 };
		vec3 zStep = { 0 };

		xStep[0] = min[0];
		yStep[0] = min[1];
		zStep[0] = min[2];

		xStep[2] = max[0];
		yStep[2] = max[1];
		zStep[2] = max[2];

		xStep[1] = xStep[0] + ((xStep[2] - xStep[0]) / 2.0f);
		yStep[1] = yStep[0] + ((yStep[2] - yStep[0]) / 2.0f);
		zStep[1] = zStep[0] + ((zStep[2] - zStep[0]) / 2.0f);

		int current[3] = { 0, 0, 0 };
		for (unsigned int c = 0; c < 8; c++) {

			current[0] = (c & 0b100) ? 1 : 0;
			current[1] = (c & 0b010) ? 1 : 0;
			current[2] = (c & 0b001) ? 1 : 0;

			children[c].min[0] = xStep[current[0]];
			children[c].min[1] = yStep[current[1]];
			children[c].min[2] = zStep[current[2]];

			children[c].max[0] = xStep[current[0]+1];
			children[c].max[1] = yStep[current[1]+1];
			children[c].max[2] = zStep[current[2]+1];

		}
	}
};

struct Vertex {
	vec4 pos;
	vec3 color;
	vec3 uv;
	vec3 normal;
	

	//NOTE: no longer used - once stored precomputed tangent/bitangent but did not work correctly
	//vec3 tangent;
	//vec3 bitangent;

	//NOTE: Required for Vulkan to use the vertex class (e.g. shaders)
	/*
	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};

		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2>
		getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_B8G8R8A8_UNORM;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_B8G8R8A8_UNORM;
		attributeDescriptions[1].offset = offsetof(Vertex, color);
		return attributeDescriptions;
	}
	*/

};