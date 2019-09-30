#pragma once
#include "linmath.h"

//TODO: handle chunking correctly (need to add chunked BVH at "count time" but currently uses pointer) 
#define BVH_CHUNK_SIZE 4000

#define BVH_BOX_SIZE 250.0f
#define VERT_IMPORT_SCALE 1000.0f

struct BVH {
	//vec3 min = { 0 };
	//vec3 max = { 0 };
	int numTris = 0;
	int triIdx[BVH_CHUNK_SIZE] = { 0 };

	int depth = 0;
	//int children[8] = { 0 };
	//int nextOctree = -1;
	int front = -1;
	int back = -1;
	bool hasTris = false;
	//float radius = 0.0f;
	vec3 maxCorner = { 0 };
	vec3 centre = { 0 };

	BVH() {
		for (int i = 0; i < BVH_CHUNK_SIZE; i++) {
			triIdx[i] = -1;
		}
		//for (int i = 0; i < 8; i++) {
		//	children[i] = -1;
		//}
	}

	BVH(const BVH &b2) {
		for (int i = 0; i < BVH_CHUNK_SIZE; i++) {
			triIdx[i] = b2.triIdx[i];
		}
		numTris = b2.numTris;
		depth = b2.depth;
		front = b2.front;
		back = b2.back;
		hasTris = b2.hasTris;
		//radius = b2.radius;
		maxCorner[0] = b2.maxCorner[0];
		maxCorner[1] = b2.maxCorner[1];
		maxCorner[2] = b2.maxCorner[2];
		centre[0] = b2.centre[0];
		centre[1] = b2.centre[1];
		centre[2] = b2.centre[2];


	}

};

struct BVH_BAKE {
	//vec3 min = { 0 };
	//vec3 max = { 0 };
	std::vector<int> triIdx;
	//std::vector<float> radii;
	float radius = 0.0f;
	vec3 centre = { 0 };
	int depth = 0;
	bool isActive = true;
	int idx = -1;
	std::vector<int> children;
	vec3 maxCorner = { 0 };

	~BVH_BAKE() {
		triIdx.clear();
		children.clear();
	}

	bool hasVerts() {
		return triIdx.size() > 0;
	}

	BVH_BAKE() {
		triIdx = std::vector<int>();
		children = std::vector<int>();
		radius = 0.0f;
		centre[0] = 0.0f;
		centre[1] = 0.0f;
		centre[2] = 0.0f;

	}

	BVH_BAKE(int depth, bool active) {
		isActive = active;
		this->depth = depth;
		radius = 0.0f;
		centre[0] = 0.0f;
		centre[1] = 0.0f;
		centre[2] = 0.0f;
	}

	BVH_BAKE(int depth, 
		//int tri,
		std::vector<int>* tris,
		vec3 maxCorner, float x, float y, float z) {
		triIdx = std::vector<int>();
		children = std::vector<int>();
		
		for (int i = 0; i < tris->size(); i++) {
			triIdx.push_back(tris->at(i));
		}
		
		//radii.push_back(radius);
		//radius = radius2;
		this->maxCorner[0] = maxCorner[0];
		this->maxCorner[1] = maxCorner[1];
		this->maxCorner[2] = maxCorner[2];

		centre[0] = x;
		centre[1] = y;
		centre[2] = z;
		this->depth = depth;
	}

	BVH_BAKE(int depth, BVH_BAKE* a, BVH_BAKE* b) {
		triIdx = std::vector<int>();
		children = std::vector<int>();

		children.push_back(a->idx);
		children.push_back(b->idx);

		this->depth = depth;
		//average children centres

		centre[0] = (
			((a->isActive) ? a->centre[0] : b->centre[0]) 
			+ ((b->isActive) ? b->centre[0] : a->centre[0])) / 2;
		centre[1] = (
			((a->isActive) ? a->centre[1] : b->centre[1]) 
			+ ((b->isActive) ? b->centre[1] : a->centre[1])) / 2;
		centre[2] = (
			((a->isActive) ? a->centre[2] : b->centre[2]) 
			+ ((b->isActive) ? b->centre[2] : a->centre[2])) / 2;


		this->maxCorner[0] = (a->maxCorner[0] + b->maxCorner[0]) / 2.0f;
		this->maxCorner[1] = (a->maxCorner[1] + b->maxCorner[1]) / 2.0f;
		this->maxCorner[2] = (a->maxCorner[2] + b->maxCorner[2]) / 2.0f;

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