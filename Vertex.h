#pragma once
#include "linmath.h"

#define BVH_CHUNK_SIZE 60
#define BVH_BOX_SIZE 250.0f
#define BVH_EXPAND_LIMIT 800.0f
#define VERT_IMPORT_SCALE 1000.0f

struct BVH {
	vec3 min = { 0 };
	vec3 max = { 0 };
	int numTris;
	int triIdx[BVH_CHUNK_SIZE] = { 0 };

	BVH() {
		for (int i = 0; i < BVH_CHUNK_SIZE; i++) {
			triIdx[i] = -1;
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
	}

};

struct Vertex {
	vec4 pos;
	vec3 color;
	vec3 uv;
	vec3 normal;
	vec3 tangent;
	vec3 bitangent;

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