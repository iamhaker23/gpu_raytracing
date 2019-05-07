/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define GLFW_INCLUDE_VULKAN

#ifdef _WIN64
#include <windows.h>
#endif

#include <glfw3.h>
#include <vulkan/vulkan.h>

#include <chrono>
#include <ctime>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#ifdef _WIN64
#include <aclapi.h>
#include <dxgi1_2.h>
#include <vulkan/vulkan_win32.h>

#include <VersionHelpers.h>
#define _USE_MATH_DEFINES
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "OBJLoader.h"

#include "linmath.h"

#define WIDTH 1024
#define HEIGHT 1024
#define SHAPE_MODE 0
#define BVH_DEBUG_VISUALISATION 0
#define CULLING 0
//slower visual presentation when CPU prints with cout (but the frames are still rendered by CUDA as fast)
#define PRINT_FPS 0

//bump mapping OR normal mapping (normal mapping will override if both set)
#define NORMAL_MAPPING 0
#define BUMP_MAPPING 1
#define BUMP_STRENGTH 0.05f

#define UV_FILTERING_ENABLED 1
#define UV_FILTER_SIZE 1000.0f

#define USE_BVH 0

 //NOTE: only support power-of-two DRSD values (for maximising GPU utilisation)
//#define DEFERRED_REFRESH_SQUARE_DIM 1
//define DEFERRED_REFRESH_SQUARE_DIM 2
#define DEFERRED_REFRESH_SQUARE_DIM 4

//Enable vulkan validation (prints Vulkan validation errors in the console window)
#define VULKAN_VALIDATION 0
//#define VULKAN_VALIDATION 1

const std::vector<const char*> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation" };

#if VULKAN_VALIDATION
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif

struct QueueFamilyIndices {
	int graphicsFamily = -1;
	int presentFamily = -1;

	bool isComplete() { return graphicsFamily >= 0 && presentFamily >= 0; }
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
	VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN64
	VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
	VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
	VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
	VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
};

#ifdef _WIN64
class WindowsSecurityAttributes {
protected:
	SECURITY_ATTRIBUTES m_winSecurityAttributes;
	PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;

public:
	WindowsSecurityAttributes();
	SECURITY_ATTRIBUTES* operator&();
	~WindowsSecurityAttributes();
};

WindowsSecurityAttributes::WindowsSecurityAttributes() {
	m_winPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(
		1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));
	// CHECK_NEQ(m_winPSecurityDescriptor, (PSECURITY_DESCRIPTOR)NULL);

	PSID* ppSID =
		(PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	InitializeSecurityDescriptor(m_winPSecurityDescriptor,
		SECURITY_DESCRIPTOR_REVISION);

	SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority =
		SECURITY_WORLD_SID_AUTHORITY;
	AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0,
		0, 0, 0, 0, 0, ppSID);

	EXPLICIT_ACCESS explicitAccess;
	ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
	explicitAccess.grfAccessPermissions =
		STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
	explicitAccess.grfAccessMode = SET_ACCESS;
	explicitAccess.grfInheritance = INHERIT_ONLY;
	explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
	explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
	explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

	SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

	SetSecurityDescriptorDacl(m_winPSecurityDescriptor, TRUE, *ppACL, FALSE);

	m_winSecurityAttributes.nLength = sizeof(m_winSecurityAttributes);
	m_winSecurityAttributes.lpSecurityDescriptor = m_winPSecurityDescriptor;
	m_winSecurityAttributes.bInheritHandle = TRUE;
}

SECURITY_ATTRIBUTES* WindowsSecurityAttributes::operator&() {
	return &m_winSecurityAttributes;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes() {
	PSID* ppSID =
		(PSID*)((PBYTE)m_winPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	if (*ppSID) {
		FreeSid(*ppSID);
	}
	if (*ppACL) {
		LocalFree(*ppACL);
	}
	free(m_winPSecurityDescriptor);
}
#endif

struct UniformBufferObject {
	mat4x4 model;
	mat4x4 view;
	mat4x4 proj;
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct Texel {
	//stbi_uc r;
	//stbi_uc g;
	//stbi_uc b;
	//stbi_uc a;

	stbi_uc col[4] = { 255 };

	Texel() {
		//this->r = 0;
		//this->g = 0;
		//this->b = 0;
		//this->a = 0;
	}

};

texture<uchar4, 2, cudaReadModeNormalizedFloat> cudaTex1;
texture<uchar4, 2, cudaReadModeNormalizedFloat> cudaTex2;
texture<uchar4, 2, cudaReadModeNormalizedFloat> cudaTex3;

struct Sphere
{
	vec3 center;                           /// position of the sphere
	float radius, radius2;                  /// sphere radius and radius^2
	vec3 surfaceColor, emissionColor;      /// surface color and emission (light)
	float transparency, reflection;         /// surface transparency and reflectivity
	bool castShadows;
	Sphere(
		const vec3 &c,
		const float &r,
		const vec3 &sc,
		const float &refl,
		const float &transp,
		const vec3 &ec,
		const bool &shadows
	) {

		center[0] = c[0];
		center[1] = c[1];
		center[2] = c[2];

		surfaceColor[0] = sc[0];
		surfaceColor[1] = sc[1];
		surfaceColor[2] = sc[2];

		emissionColor[0] = ec[0];
		emissionColor[1] = ec[1];
		emissionColor[2] = ec[2];

		castShadows = shadows;

		transparency = transp;
		reflection = refl;

		radius = r;
		radius2 = r * r;

	}

};

__device__ float magnitude(vec3 v) {
	float total = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	return total;

}

__device__ float dot(vec3 l, vec3 r) {
	return (l[0] * r[0]) + (l[1] * r[1]) + (l[2] * r[2]);
}


__device__ float dotAxis(vec3 l, int axis, float dir) {

	return 
		(l[0] * ((axis ==0)? dir : 0))
		+ (l[1] * ((axis == 1) ? dir : 0))
		+ (l[2] * ((axis == 2) ? dir : 0));

}

__device__ float cross(int axis, vec3 a, vec3 b) {
	//cx = aybz - azby
	if (axis == 0) return a[1] * b[2] - a[2] * b[1];
	//cx = azbx - axbz
	if (axis == 1) return a[2] * b[0] - a[0] * b[2];
	//cx = axby - aybx
	if (axis == 2) return a[0] * b[1] - a[1] * b[0];

	//unsupported axis index?
	return 0.0f;
}

__device__ float intersect(Sphere sphere, vec3 &rayorig, vec3 &raydir)
{
	vec3 l = { 0 };
	l[0] = sphere.center[0] - rayorig[0];
	l[1] = sphere.center[1] - rayorig[1];
	l[2] = sphere.center[2] - rayorig[2];

	float tca = dot(l, raydir);
	if (tca < 0) return -1.0f;
	float d2 = dot(l, l) - tca * tca;
	if (d2 > sphere.radius2) return -1.0f;
	float thc = sqrt(sphere.radius2 - d2);

	float tmp1 = tca - thc;
	float tmp2 = tca + thc;

	if (tmp1 < 0) return tmp2;
	return tmp1;
}

__device__ float hasIntersection(Sphere sphere, vec3 &rayorig, vec3 &raydir, float maxDist)
{
	vec3 l = { 0 };
	l[0] = sphere.center[0] - rayorig[0];
	l[1] = sphere.center[1] - rayorig[1];
	l[2] = sphere.center[2] - rayorig[2];

	//Shadow-caster is on the other side of the light, no shadow.
	if (magnitude(l) - sphere.radius > maxDist) return false;

	float tca = dot(l, raydir);
	if (tca < 0) return false;
	float d2 = dot(l, l) - tca * tca;
	if (d2 > sphere.radius2) return false;

	return true;
}

struct IntersectionResult{
	/*
	vec4 pos = { 0 };
	vec4 normal = { 0 };
	*/
	float tmin = -1.0f;
	vec3 pos = { 0 };
	
	vec3 normal = { 0 };
	//vec3 tangent;
	//vec3 bitangent;

	vec3 uv = { 0 };
	vec4 col = { 0 };
	int triIndex = -1;
	int faceDir = 0;
};

size_t mesh_width = 0, mesh_height = 0;
std::string execution_path;

////START RAYTRACING


//const int MAX_RAY_DEPTH = 3;
const int MAX_RAY_DEPTH = 4;

__device__ void blendCol(Texel* col, float factor, float r, float g, float b) {

	//normalise light ("HDR-like")
	float largest = b;
	if (r > g) {
		if (r > b) {
			largest = r;
		}
	}
	else {
		if (g > b) {
			largest = g;
		}
	}

	float lightOverspill = largest - 255;
	if (lightOverspill < 0) lightOverspill = 0;
	/*
	col->g = (col->g  * (1.0f - factor)) + (factor * (g- lightOverspill));
	col->b = (col->b * (1.0f - factor)) + (factor * (b-lightOverspill));
	col->r = (col->r * (1.0f - factor)) + (factor *  (r-lightOverspill));
	*/


	col->col[0] = (col->col[0] * (1.0f - factor)) + (factor * (b - lightOverspill));
	col->col[1] = (col->col[1] * (1.0f - factor)) + (factor * (g - lightOverspill));
	col->col[2] = (col->col[2] * (1.0f - factor)) + (factor *  (r - lightOverspill));

	//col->col[3] = 0xFFFFFFFF;

}

__device__ void setCol(Texel* col, float r, float g, float b) {

	//TODO: b8g8r8a8_unorm format supported on 

	col->col[0] = b;
	col->col[1] = g;
	col->col[2] = r;
	//col->col[3] = 0xFFFFFFFF;
}

__device__ void intersectTris(int* vertIdx, Vertex* verts, IntersectionResult* outhit, float factor, int numTris, float orig_x, float orig_y, float orig_z, float ray_x, float ray_y, float ray_z, float maxDist) {
		
	float minU = -1.0f;
	float minV = -1.0f;
	float minT = 99999999.0f;
	int faceDir = 0;
	int closestTri = -1;

	vec3 raydir = { 0 };
	raydir[0] = ray_x;
	raydir[1] = ray_y;
	raydir[2] = ray_z;
	float Epsilon = 0.001f;

	// BVH steps through triangle list but no BVH steps directly through vertex list
#if USE_BVH == 0

	//BVH resets outhit, if no bvh it must be done here
	outhit->pos[0] = 0;
	outhit->pos[1] = 0;
	outhit->pos[2] = 0;
	outhit->col[0] = 0;
	outhit->col[1] = 0;
	outhit->col[2] = 0;
	outhit->normal[0] = 0;
	outhit->normal[1] = 0;
	outhit->normal[2] = 0;
	outhit->triIndex = -1;
	outhit->faceDir = 0;
	outhit->tmin = -1;
	outhit->uv[0] = 0;
	outhit->uv[1] = 0;


	int step = 3;
#else
	int step = 1;
#endif

	//triangle - 3 index of verts
	//triangle - 3 index in bvh index list

	for (int tri = 0; tri < numTris*step; tri += step) {
		//Moller-Trumbore
		//V1
		//V2
		//V3
		//RayOrigin
		//RayDir

		vec3 V1 = { 0 };
		vec3 V2 = { 0 };
		vec3 V3 = { 0 };


		//Order is 2,1,0 to reverse face loop
		//For correct UV mapping
#if USE_BVH == 0

		V1[0] = verts[tri + 2].pos[0];
		V1[1] = verts[tri + 2].pos[1];
		V1[2] = verts[tri + 2].pos[2];

		V2[0] = verts[tri + 1].pos[0];
		V2[1] = verts[tri + 1].pos[1];
		V2[2] = verts[tri + 1].pos[2];

		V3[0] = verts[tri + 0].pos[0];
		V3[1] = verts[tri + 0].pos[1];
		V3[2] = verts[tri + 0].pos[2];
#else

		V1[0] = verts[vertIdx[tri]+2].pos[0];
		V1[1] = verts[vertIdx[tri]+2].pos[1];
		V1[2] = verts[vertIdx[tri]+2].pos[2];

		V2[0] = verts[vertIdx[tri]+1].pos[0];
		V2[1] = verts[vertIdx[tri]+1].pos[1];
		V2[2] = verts[vertIdx[tri]+1].pos[2];

		V3[0] = verts[vertIdx[tri]+0].pos[0];
		V3[1] = verts[vertIdx[tri]+0].pos[1];
		V3[2] = verts[vertIdx[tri]+0].pos[2];
#endif

		/* //IF ALL VERTS ARE BEHIND CAMERA
		if (V1[2] > 0) V1[2] = -1.0f;
		if (V2[2] > 0) V2[2] = -1.0f;
		if (V3[2] > 0) V3[2] = -1.0f;
		if (V1[2] == -1.0f && V2[2] == -1.0f && V3[2] == -1.0f) continue;
		*/

		//Edge1 = V2-v1
		//Edge2 = V3-V1
		//h=raydir.cross(Edge2)
		//A=Edge1.dot(h)
		//IF A < Epsilon &&  A > -Epsilon (no hit)

		vec3 edge1 = { 0 };
		edge1[0] = V2[0] - V1[0];
		edge1[1] = V2[1] - V1[1];
		edge1[2] = V2[2] - V1[2];

		vec3 edge2 = { 0 };
		edge2[0] = V3[0] - V1[0];
		edge2[1] = V3[1] - V1[1];
		edge2[2] = V3[2] - V1[2];

		vec3 h = { 0 };
		h[0] = cross(0, raydir, edge2);
		h[1] = cross(1, raydir, edge2);
		h[2] = cross(2, raydir, edge2);

		float A = dot(edge1, h);

#ifdef CULLING 
		if (A < Epsilon) continue;
#endif
		//nohit
		if (A < Epsilon && A > -Epsilon) continue;
		
		//s=rayOrigin-V1
		//U=s.dot(h)/A
		//IF U < 0.0 || U > 1.0 (no hit)

		vec3 s = { 0 };
		s[0] = orig_x - V1[0];
		s[1] = orig_y - V1[1];
		s[2] = orig_z - V1[2];

		float U = dot(s, h) / A;

		if (U < 0.0f || U > 1.0f) continue;

		//Q=s.cross(Edge1)
		//V=raydir.dot(Q)/A
		//IF V < 0.0 || U+V > 1.0 (no hit)

		vec3 Q = { 0 };
		Q[0] = cross(0, s, edge1);
		Q[1] = cross(1, s, edge1);
		Q[2] = cross(2, s, edge1);

		float V = dot(raydir, Q) / A;

		if (V < 0.0 || U + V > 1.0) continue;

		//t=edge2.dot(Q)/A
		//IF t > Epsilon (hit, return RayOrigin+RayDir*t)
		//ELSE no hit
		float t = dot(edge2, Q) / A;
		if (t > Epsilon) {

			if (t < minT && (maxDist < 0 || t < maxDist) 
				//When using BVH we must check that this triangle intersection is in front of any earlier drawn tris
				&& (outhit->tmin < 0.0f || t < outhit->tmin) ) 
			{
				minT = t;
				minU = U;
				minV = V;

#if USE_BVH == 0
				closestTri = tri;
#else
				closestTri = vertIdx[tri];
#endif

				faceDir = (A < Epsilon) ? -1 : 1;
			}
		}
	}

	if (closestTri >= 0){
		outhit->pos[0] = orig_x + (minT * ray_x);
		outhit->pos[1] = orig_y + (minT * ray_y);
		outhit->pos[2] = orig_z + (minT * ray_z);

		outhit->col[0] = 
			(verts[closestTri + 0].color[0] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].color[0] * minV) +
			(verts[closestTri + 2].color[0] * minU);

		outhit->col[1] = 
			(verts[closestTri + 0].color[1] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].color[1] * minV) +
			(verts[closestTri + 2].color[1] * minU);

		outhit->col[2] = 
			(verts[closestTri + 0].color[2] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].color[2] * minV) +
			(verts[closestTri + 2].color[2] * minU);

		outhit->normal[0] =
			(verts[closestTri + 0].normal[0] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].normal[0] * minV) +
			(verts[closestTri + 2].normal[0] * minU);
		outhit->normal[1] =
			(verts[closestTri + 0].normal[1] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].normal[1] * minV) +
			(verts[closestTri + 2].normal[1] * minU);
		outhit->normal[2] =
			(verts[closestTri + 0].normal[2] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].normal[2] * minV) +
			(verts[closestTri + 2].normal[2] * minU);

		//bugfix -> Blender UV co-ordinate system has flipped U compared to tex2D
#if UV_FILTERING_ENABLED == 1
		
			//filtersize blurs normal map (to quantize interpolation and reduce noise)
		outhit->uv[0] = roundf((
			(verts[closestTri + 0].uv[0] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].uv[0] * minV) +
			(verts[closestTri + 2].uv[0] * minU)
			)*UV_FILTER_SIZE) / UV_FILTER_SIZE;
		outhit->uv[1] = roundf((1.0f - (
			(verts[closestTri + 0].uv[1] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].uv[1] * minV) +
			(verts[closestTri + 2].uv[1] * minU)
			))*UV_FILTER_SIZE) / UV_FILTER_SIZE;

#else
		//filtersize blurs normal map (to quantize interpolation and reduce noise)
		outhit->uv[0] = (
			(verts[closestTri + 0].uv[0] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].uv[0] * minV) +
			(verts[closestTri + 2].uv[0] * minU)
			) ;
		outhit->uv[1] = ((1.0f - (
			(verts[closestTri + 0].uv[1] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].uv[1] * minV) +
			(verts[closestTri + 2].uv[1] * minU)
			)));
#endif
		//float uvmag = magnitude(outhit->uv);
		//outhit->uv[0] = outhit->uv[0] / WIDTH;
		//outhit->uv[1] = outhit->uv[1] / HEIGHT;

		//outhit->normal[0] = -outhit->normal[0];
		//outhit->normal[1] = -outhit->normal[1];
		//outhit->normal[2] = -outhit->normal[2];

		outhit->triIndex = closestTri;
		outhit->faceDir = faceDir;
		outhit->tmin = minT;

		//tangent axis is stored in first vertex only
		//outhit->tangent[0] = verts[closestTri + 0].tangent[0];
		//outhit->tangent[1] = verts[closestTri + 0].tangent[1];
		//outhit->tangent[2] = verts[closestTri + 0].tangent[2];

		//bitangent axis is stored in first vertex only
		//outhit->bitangent[0] = verts[closestTri + 0].bitangent[0];
		//outhit->bitangent[1] = verts[closestTri + 0].bitangent[1];
		//outhit->bitangent[2] = verts[closestTri + 0].bitangent[2];


	}
	/*
		outhit->pos[0] =
			(verts[closestTri + 0].pos[0] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].pos[0] * minV) +
			(verts[closestTri + 2].pos[0] * minU)
			;
		outhit->pos[1] =
			(verts[closestTri + 0].pos[1] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].pos[1] * minV) +
			(verts[closestTri + 2].pos[1] * minU)
			;
		outhit->pos[2] =
			(verts[closestTri + 0].pos[2] * (1.0 - minV - minU)) +
			(verts[closestTri + 1].pos[2] * minV) +
			(verts[closestTri + 2].pos[2] * minU)
			;
	*/
}

template<int depth>
__device__ void intersectBVH(BVH* bvh
	, int numBoxes
	, Vertex* verts
	, IntersectionResult* outhit
	, float factor
	, int numTris
	, float orig_x, float orig_y, float orig_z
	, float ray_x, float ray_y, float ray_z
	, float maxDist) {
	
	if (depth == 1) {
		outhit->pos[0] = 0;
		outhit->pos[1] = 0;
		outhit->pos[2] = 0;
		outhit->col[0] = 0;
		outhit->col[1] = 0;
		outhit->col[2] = 0;
		outhit->normal[0] = 0;
		outhit->normal[1] = 0;
		outhit->normal[2] = 0;
		outhit->triIndex = -1;
		outhit->faceDir = 0;
		outhit->tmin = -1;
		outhit->uv[0] = 0;
		outhit->uv[1] = 0;
	}

	vec3 raydir = { 0 };
	raydir[0] = ray_x;
	raydir[1] = ray_y;
	raydir[2] = ray_z;
	float raydirmag = magnitude(raydir);
	raydir[0] = 1.0f / raydir[0];
	raydir[1] = 1.0f / raydir[1];
	raydir[2] = 1.0f / raydir[2];

	//float minT = 99999.9f;

	int nextOctree = -1;
	//backwards because the octree is biased in the negative direction
	//TODO: this would cause depth failures when facing the other direction!
	for (int i = 0; i <= numBoxes-1; i++) {

		//PROBLEM: where are the root octrees? Sparsely populated...
		//TRY: next octree in each BVH
		//NOTE: ensures currentOctree==i when depth>1
		//NOTE2: must use nextOctree regardless of level!
		
		int currentOctree = -1;
		if (depth == 1) {
			currentOctree = (nextOctree == -1) ? i : nextOctree;
			if (currentOctree == -1) return;
			nextOctree = bvh[currentOctree].nextOctree;
		}
		else {
			currentOctree = bvh->children[i];
			if (currentOctree == -1) continue;
		}

		//for each root octree
		//recurse until hit at lowest level hit
		//Parent -> 
		//child1 -> grandchildren -> ... 
		//child2 -> grandchildren -> ...
		// ...
		//child8 -> grandchildren -> ...

		vec3 t0s = { 0 };
		t0s[0] = ((bvh[currentOctree].min[0] - (orig_x)) * (raydir[0]));
		t0s[1] = ((bvh[currentOctree].min[1] - (orig_y)) * (raydir[1]));
		t0s[2] = ((bvh[currentOctree].min[2] - (orig_z)) * (raydir[2]));

		vec3 t1s = { 0 };
		t1s[0] = ((bvh[currentOctree].max[0] - (orig_x)) * (raydir[0]));
		t1s[1] = ((bvh[currentOctree].max[1] - (orig_y)) * (raydir[1]));
		t1s[2] = ((bvh[currentOctree].max[2] - (orig_z)) * (raydir[2]));

		float tmin = max(min(t0s[0], t1s[0]), max(min(t0s[1], t1s[1]), min(t0s[2], t1s[2])));
		float tmax = min(max(t0s[0], t1s[0]), min(max(t0s[1], t1s[1]), max(t0s[2], t1s[2])));

		if (tmin < tmax) {

			//iterate children
			intersectBVH<depth + 1>(&bvh[currentOctree]
				, 8
				, verts
				, outhit
				, factor
				, numTris
				, orig_x, orig_y, orig_z
				, ray_x, ray_y, ray_z
				, maxDist);

			//if (outhit->triIndex != -1) return;

		}
	}
}

template<>
__device__ void intersectBVH<MAX_BVH_DEPTH>(BVH* bvh
	, int numBoxes
	, Vertex* verts
	, IntersectionResult* outhit
	, float factor
	, int numTris
	, float orig_x, float orig_y, float orig_z
	, float ray_x, float ray_y, float ray_z
	, float maxDist) {

	vec3 raydir = { 0 };
	raydir[0] = ray_x;
	raydir[1] = ray_y;
	raydir[2] = ray_z;
	float raydirmag = magnitude(raydir);
	raydir[0] = 1.0f / raydir[0];
	raydir[1] = 1.0f / raydir[1];
	raydir[2] = 1.0f / raydir[2];

	//float minT = 99999.9f;

	int nextOctree = -1;
	//backwards because the octree is biased in the negative direction
	//TODO: this would cause depth failures when facing the other direction!
	for (int i = 0; i <= numBoxes - 1; i++) {

		//PROBLEM: where are the root octrees? Sparsely populated...
		//TRY: next octree in each BVH
		//NOTE: ensures currentOctree==i when depth>1
		//NOTE2: must use nextOctree regardless of level!

		int currentOctree = -1;
		if (MAX_BVH_DEPTH == 1) {
			currentOctree = (nextOctree == -1) ? i : nextOctree;
			if (currentOctree == -1) return;
			nextOctree = bvh[currentOctree].nextOctree;
		}
		else {
			currentOctree = bvh->children[i];
			if (currentOctree == -1) continue;
		}

		//for each root octree
		//recurse until hit at lowest level hit
		//Parent -> 
		//child1 -> grandchildren -> ... 
		//child2 -> grandchildren -> ...
		// ...
		//child8 -> grandchildren -> ...

		vec3 t0s = { 0 };
		t0s[0] = ((bvh[currentOctree].min[0] - (orig_x)) * (raydir[0]));
		t0s[1] = ((bvh[currentOctree].min[1] - (orig_y)) * (raydir[1]));
		t0s[2] = ((bvh[currentOctree].min[2] - (orig_z)) * (raydir[2]));

		vec3 t1s = { 0 };
		t1s[0] = ((bvh[currentOctree].max[0] - (orig_x)) * (raydir[0]));
		t1s[1] = ((bvh[currentOctree].max[1] - (orig_y)) * (raydir[1]));
		t1s[2] = ((bvh[currentOctree].max[2] - (orig_z)) * (raydir[2]));

		float tmin = max(min(t0s[0], t1s[0]), max(min(t0s[1], t1s[1]), min(t0s[2], t1s[2])));
		float tmax = min(max(t0s[0], t1s[0]), min(max(t0s[1], t1s[1]), max(t0s[2], t1s[2])));

		if (tmin < tmax) {

			//debug the BVH
#if BVH_DEBUG_VISUALISATION == 1
			outhit->col[0] = 1;
			outhit->col[1] = 1;
			outhit->col[2] = 0;
#else		
			//check triangles for any hit BVH
			//Can't ignore further BVH boxes in case of a BVH hit but Tri miss
			intersectTris(
				bvh[currentOctree].triIdx
				, verts
				, outhit
				, factor
				, bvh[currentOctree].numTris
				, orig_x, orig_y, orig_z
				, ray_x, ray_y, ray_z
				, maxDist);
#endif
			//Can do this when BVH are sorted (e.g. first hit is certain to be front-most
			//if (outhit->triIndex >= 0) return;
		}
	}
}

#if SHAPE_MODE==0

template<int depth>
__device__ void RaytraceTris(
	Texel* col
	, float factor
	, BVH* bvh
	, int numBVH
	, Vertex* verts
	, int numTris
	, float ray_x, float ray_y, float ray_z
	, float orig_x, float orig_y, float orig_z
	, float light_x, float light_y, float light_z
	, IntersectionResult* hitlist
	, int light_mode
	, float x1, float y1)
{


#if USE_BVH == 0
	int null = 0;
	intersectTris(&null, verts, hitlist, factor, numTris, orig_x, orig_y, orig_z, ray_x, ray_y, ray_z, -1.0f);
#else
	intersectBVH<1>(bvh, numBVH, verts, hitlist, factor, numTris, orig_x, orig_y, orig_z, ray_x, ray_y, ray_z, -1.0f);
#endif

	int tri = hitlist->triIndex;
	bool inside = true;// (hitlist->faceDir == -1) ? 1 : 0;

	if (tri < 0) {
		//miss - background colour (or intersection debug color)
		if (hitlist->col[0] + hitlist->col[1] + hitlist->col[2] == 0) {
			
			//blendCol(col, factor, 150, 160, 200);

			float4 bgTex = tex2D(cudaTex3, (x1 +(WIDTH/2))/WIDTH, (y1 + (HEIGHT / 2)) / HEIGHT);

			blendCol(col, factor
				, 255 * bgTex.x
				, 255 * bgTex.y
				, 255 * bgTex.z
			);
		}
		else {
			blendCol(col, factor, 255*hitlist->col[0], 255 * hitlist->col[1], 255 * hitlist->col[2]);
		}
	}
	else {

		vec3 hitPos = { 0 };
		hitPos[0] = hitlist->pos[0];
		hitPos[1] = hitlist->pos[1];
		hitPos[2] = hitlist->pos[2];
		vec3 diffCol = { 0 };
		diffCol[0] = hitlist->col[0];
		diffCol[1] = hitlist->col[1];
		diffCol[2] = hitlist->col[2];

		//Tex2D get colors as .xyz
		float4 diffTex = tex2D(cudaTex1, hitlist->uv[0], hitlist->uv[1]);
		diffCol[0] = diffCol[0] * diffTex.x;
		diffCol[1] = diffCol[1] * diffTex.y;
		diffCol[2] = diffCol[2] * diffTex.z;
		

		//float normalIntensity = 2.0f;
		vec3 nnhit = { 0 };
		//vec3 geometricNor = { 0 };
		nnhit[0] = hitlist->normal[0];
		nnhit[1] = hitlist->normal[1];
		nnhit[2] = hitlist->normal[2];

		float nnhitmag = magnitude(nnhit);
		nnhit[0] = nnhit[0] / nnhitmag;
		nnhit[1] = nnhit[1] / nnhitmag;
		nnhit[2] = nnhit[2] / nnhitmag;

		vec3 raydir = { 0 };
		raydir[0] = ray_x;
		raydir[1] = ray_y;
		raydir[2] = ray_z;
		float raydirmag = magnitude(raydir);
		raydir[0] = ray_x / raydirmag;
		raydir[1] = ray_y / raydirmag;
		raydir[2] = ray_z / raydirmag;


		vec3 nraydir = { 0 };
		//nraydir[0] = 1.0 / (ray_x+1.0f);
		//nraydir[1] = 1.0 / (ray_y + 1.0f);
		//nraydir[2] = 1.0 / (ray_z + 1.0f);

		nraydir[0] = -raydir[0];// / raydirmag;
		nraydir[1] = -raydir[1];// / raydirmag;
		nraydir[2] = -raydir[2];// / raydirmag;

		float cosi = dot(nnhit, nraydir);
		if (cosi > 0) {
			inside = false;
			nnhit[0] *= -1.0f;
			nnhit[1] *= -1.0f;
			nnhit[2] *= -1.0f;
		}
		else {
			cosi = -cosi;
		}

		vec3 lightPos = { 0 };
		lightPos[0] = light_x;
		lightPos[1] = light_y;
		lightPos[2] = light_z;

		vec3 toLight = { 0 };
		toLight[0] = lightPos[0] - (hitPos[0]);
		toLight[1] = lightPos[1] - (hitPos[1]);
		toLight[2] = lightPos[2] - (hitPos[2]);
		float distToLight = magnitude(toLight);
		toLight[0] = toLight[0] / distToLight;
		toLight[1] = toLight[1] / distToLight;
		toLight[2] = toLight[2] / distToLight;
		
		float normalIntensity = BUMP_STRENGTH;

#if NORMAL_MAPPING == 1
		//tangent space normal (i.e. oriented to triangle)
		
		vec3 nnhitTang = { 0 };

		float4 normTex = tex2D(cudaTex2, hitlist->uv[0], hitlist->uv[1]);
		nnhitTang[0] = (normTex.x * 2.0 - 1.0);
		nnhitTang[1] = (normTex.y * 2.0 - 1.0);
		nnhitTang[2] = -normalIntensity * (normTex.z * 2.0 - 1.0);
		//Convert tangent-space normal map to world-space
		nnhit[0] = dotAxis(nnhitTang, 0, 1);
		nnhit[1] = dotAxis(nnhitTang, 1, 1);
		nnhit[2] = dotAxis(nnhitTang, 2, 1);
		float nh = magnitude(nnhit);
		nnhit[0] /= nh;
		nnhit[1] /= nh;
		nnhit[2] /= nh;

#elif BUMP_MAPPING == 1
		//tangent space normal (i.e. oriented to triangle)
		
		vec3 nnhitTang = { 0 };

		float4 normTex = tex2D(cudaTex2, hitlist->uv[0], hitlist->uv[1]);
		nnhitTang[0] = (normTex.x * 2.0 - 1.0);
		nnhitTang[1] = (normTex.y * 2.0 - 1.0);
		nnhitTang[2] = normalIntensity * (normTex.z * 2.0 - 1.0);
		//Convert tangent-space bump map to world-space
		nnhitTang[0] = dotAxis(nnhitTang, 0, 1);
		nnhitTang[1] = dotAxis(nnhitTang, 1, 1);
		nnhitTang[2] = dotAxis(nnhitTang, 2, 1);
		float nh = magnitude(nnhitTang);
		nnhitTang[0] /= nh;
		nnhitTang[1] /= nh;
		nnhitTang[2] /= nh;
		//add bump normal to geometry normal
		nnhit[0] = ((1.0f - normalIntensity)*nnhit[0]) - (normalIntensity*nnhitTang[0]);
		nnhit[1] = ((1.0f - normalIntensity)*nnhit[1]) - (normalIntensity*nnhitTang[1]);
		nnhit[2] = ((1.0f - normalIntensity)*nnhit[2]) - (normalIntensity*nnhitTang[2]);
		//renormalize
		float nh2 = magnitude(nnhit);
		nnhit[0] /= nh2;
		nnhit[1] /= nh2;
		nnhit[2] /= nh2;

#endif
		//use normal map value directly
		//nnhit[0] = nnhitTang[0];
		//nnhit[1] = nnhitTang[1];
		//nnhit[2] = nnhitTang[2];
		
		/*
		//toLight in tri tangent-space
		vec3 toLightTang = { 0 };
		toLightTang[0] = toLight[0];
		toLightTang[1] = toLight[1];
		toLightTang[2] = toLight[2];
		toLightTang[0] = dotAxis(toLightTang, 0, nnhit[0]);
		toLightTang[1] = dotAxis(toLightTang, 1, nnhit[1]);
		toLightTang[2] = dotAxis(toLightTang, 2, nnhit[2]);
		float distToLightTang = magnitude(toLightTang);
		toLightTang[0] = toLightTang[0] / distToLightTang;
		toLightTang[1] = toLightTang[1] / distToLightTang;
		toLightTang[2] = toLightTang[2] / distToLightTang;
		*/

		vec3 nnnhit = { 0 };
		nnnhit[0] = -nnhit[0];
		nnnhit[1] = -nnhit[1];
		nnnhit[2] = -nnhit[2];

#if USE_BVH == 0
		intersectTris(&null, verts, hitlist, factor, numTris
			, hitPos[0] + nnhit[0] * 1e-2
			, hitPos[1] + nnhit[1] * 1e-2
			, hitPos[2] + nnhit[2] * 1e-2
			, toLight[0], toLight[1], toLight[2]
			, distToLight);
#else
		intersectBVH<1>(bvh, numBVH, verts, hitlist, factor, numTris
			, hitPos[0] + nnhit[0] * 1e-2
			, hitPos[1] + nnhit[1] * 1e-2
			, hitPos[2] + nnhit[2] * 1e-2
			, toLight[0], toLight[1], toLight[2]
			, distToLight);
#endif

		int shadowCaster = hitlist->triIndex;

		float m = 0.1f;

		// no shadow caster, do shading
		if (shadowCaster == -1 || shadowCaster == tri) {

			/*
			vec3 nnnhitTang = { 0 };
			nnnhitTang[0] = -nnhitTang[0];
			nnnhitTang[1] = -nnhitTang[1];
			nnnhitTang[2] = -nnhitTang[2];
			*/

			//Fix: removed double-sided lighting
			m = max(m,
				// max(
				//	(dot(toLight, nnhit))
					 (dot(toLight, nnnhit))
				//)
			);
			
			/*
			vec3 raydirTang = { 0 };
			raydirTang[0] = raydir[0];
			raydirTang[1] = raydir[1];
			raydirTang[2] = raydir[2];
			raydirTang[0] = dotAxis(raydirTang, 0, nnhit[0]);
			raydirTang[1] = dotAxis(raydirTang, 1, nnhit[1]);
			raydirTang[2] = dotAxis(raydirTang, 2, nnhit[2]);
			float drdt = magnitude(raydirTang);
			raydirTang[0] = raydirTang[0] / drdt;
			raydirTang[1] = raydirTang[1] / drdt;
			raydirTang[2] = raydirTang[2] / drdt;
			*/

			vec3 halfwayDir = { 0 };
			halfwayDir[0] = toLight[0] * distToLight + raydir[0] * raydirmag;
			halfwayDir[1] = toLight[1] * distToLight  + raydir[1] * raydirmag;
			halfwayDir[2] = toLight[2] * distToLight + raydir[2] * raydirmag;
			float hwdmag = magnitude(halfwayDir);
			halfwayDir[0] /= hwdmag;
			halfwayDir[1] /= hwdmag;
			halfwayDir[2] /= hwdmag;

			//TODO: variable shininess using pow and exponent
			float spec = max(
				abs(dot(halfwayDir, nnhit))
				, abs(dot(halfwayDir, nnnhit))
			);

			//light "power"
			m = m * 1.5f;
			m = max(0.1f, (0.5f*(spec * spec * spec * spec)*m) + (0.5f*m));
			m = min(1.0f, m);
		}

		/*
		setCol(col
			, 255 * diffCol[0] * m
			, 255 * diffCol[1] * m
			, 255 * diffCol[2] * m);
			*/

		//vec3 reflCol = { 0 };
		//vec3 refrCol = { 0 };
		
		//float facingratio = cosi;// dot(nraydir, nnhit);
		//float fresneleffect = (light_mode == 0) ? 1.0f : 0.5f;
		//float fresneleffect = max(0.1f, (0.9f * (1 - facingratio) * (1 - facingratio) * (1 - facingratio)));
		//fresneleffect = min(fresneleffect, 0.5f);

		//calculate fresnel in case of transp
		float fresneleffect = 0.0f;
		if (light_mode >= 2) {
			//more transparent the more aligned normal is to reverse raydir
			float fac = dot(raydir, nnnhit);
			fresneleffect = max(0.1f, min(0.4f, 0.5f*((1 - fac) * (1 - fac)*(0.9f)) ));
		}

		//if (light_mode >= 2) {

			float eta = (inside) ? 1.01f : 1.0f / 1.01f;

			vec3 refrdir = { 0 };

			//refrdir[0] = (k < 0) ? 0 : (eta * raydir[0]) + (eta * cosi - sqrtf(k)) * nnhit[0];
			//refrdir[1] = (k < 0) ? 0 : (eta * raydir[1]) + (eta * cosi - sqrtf(k)) * nnhit[1];
			//refrdir[2] = (k < 0) ? 0 : (eta * raydir[2]) + (eta * cosi - sqrtf(k)) * nnhit[2];


			//ACTUAL REFRACTION
			float k = max(0.0f, 1 - eta * eta * (1 - cosi * cosi));
			refrdir[0] = (raydir[0] * eta) + (nnnhit[0] * ((eta * cosi ) - (k)));
			refrdir[1] = (raydir[1] * eta) + (nnnhit[1] * ((eta * cosi ) - (k)));
			refrdir[2] = (raydir[2] * eta) + (nnnhit[2] * ((eta * cosi ) - (k)));
			//This does straight-through transparency
			//refrdir[0] = (raydir[0] );
			//refrdir[1] = (raydir[1] );
			//refrdir[2] = (raydir[2] );

			//normalise (and invert y)
			float refrMag = magnitude(refrdir);
			refrdir[0] = refrdir[0] / refrMag;
			refrdir[1] = refrdir[1] / refrMag;
			refrdir[2] = refrdir[2] / refrMag;

			/*
			intersectTris(2, verts, hitlist, factor, numTris
				, hitPos[0] + hitlist->normal[0] * 1e-2
				, hitPos[1] + hitlist->normal[1] * 1e-2
				, hitPos[2] + hitlist->normal[2] * 1e-2
				, refrdir[0], refrdir[1], refrdir[2]
				, -1.0f);
				*/

			/*OLD REFR
			intersectBVH(bvh, numBVH, verts, hitlist, factor, numTris
				, hitPos[0] + (nnhit[0] * 1e-2)
				, hitPos[1] + (nnhit[1] * 1e-2)
				, hitPos[2] + (nnhit[2] * 1e-2)
				, refrdir[0], refrdir[1], refrdir[2]
				, -1.0f);

			refrCol[0] = hitlist->col[0];
			refrCol[1] = hitlist->col[1];
			refrCol[2] = hitlist->col[2];
			*/

			/*
			blendCol(col, (1 - fresneleffect)
				, 255 * (refrCol[0] * transp)
				, 255 * (refrCol[1] * transp)
				, 255 * (refrCol[2] * transp)
			);
			*/

		//}

		//if (light_mode == 1 || light_mode == 3) {

			float cosi2 = dot(raydir, nnhit);

			vec3 refldir = { 0 };
			refldir[0] = raydir[0] - (nnhit[0] * 2 * cosi2);
			refldir[1] = raydir[1] - (nnhit[1] * 2 * cosi2);
			refldir[2] = raydir[2] - (nnhit[2] * 2 * cosi2);

			float refldirmag = magnitude(refldir);
			refldir[0] = refldir[0] / refldirmag;
			refldir[1] = refldir[1] / refldirmag;
			refldir[2] = refldir[2] / refldirmag;

			/*
			intersectTris(2, verts, hitlist, factor, numTris
				, hitPos[0] + hitlist->normal[0] * 1e-2
				, hitPos[1] + hitlist->normal[1] * 1e-2
				, hitPos[2] + hitlist->normal[2] * 1e-2
				, refldir[0], refldir[1], refldir[2]
				, -1.0f);
				*/

			/*OLD REFL
			intersectBVH(bvh, numBVH, verts, hitlist, factor, numTris
				, hitPos[0] + nnhit[0] * 1e-2
				, hitPos[1] + nnhit[1] * 1e-2
				, hitPos[2] + nnhit[2] * 1e-2
				, refldir[0], refldir[1], refldir[2]
				, -1.0f);

			reflCol[0] = hitlist->col[0];
			reflCol[1] = hitlist->col[1];
			reflCol[2] = hitlist->col[2];
			*/

			/*
			blendCol(col, fresneleffect
				, 255 * (reflCol[0] * fresneleffect)
				, 255 * (reflCol[1] * fresneleffect)
				, 255 * (reflCol[2] * fresneleffect)
			);
			*/
		//}

		//diffuse to refl/refr ratio
		//TODO:
		//at the final bounce we'll collect background color, should use russian roulette here!
		float diffuseBlend = (depth + 1 == MAX_RAY_DEPTH) ? 0.9f : 0.5f;

		diffCol[0] = diffCol[0] * m;
		diffCol[1] = diffCol[1] * m;
		diffCol[2] = diffCol[2] * m;

		if (depth == 1 && factor > 0.99f) {
			setCol(col
				, 255 * diffCol[0]
				, 255 * diffCol[1]
				, 255 * diffCol[2]);
		}
		else {
			blendCol(col, factor*diffuseBlend
				, 255 * diffCol[0]
				, 255 * diffCol[1]
				, 255 * diffCol[2]);
		}

		//we have some transp/refl to calculate
		if (light_mode != 0 && 1.0f - diffuseBlend > 0.0f) {


			//TODO: recursive raytrace depth
			//raytrace refl
			//factor = (1.0f - diffuseBlend) * fresneleffect
			if (light_mode == 1 || light_mode == 3) RaytraceTris<depth+1>(col
				, (1.0f - diffuseBlend) * (0.5f - fresneleffect)
				, bvh
				, numBVH
				, verts
				, numTris
				, refldir[0], refldir[1], refldir[2]
				, hitPos[0] + nnhit[0] * 1e-2, hitPos[1] + nnhit[1] * 1e-2, hitPos[2] + nnhit[2] * 1e-2
				, light_x, light_y, light_z
				, hitlist
				, light_mode
				, x1, y1);
			//raytrace refr
			//factor = (1.0f - diffuseBlend) * (1 - fresneleffect) * transp
			if (light_mode >= 2) RaytraceTris<depth+1>(col
				, (1.0f - diffuseBlend) * fresneleffect
				, bvh
				, numBVH
				, verts
				, numTris
				, refrdir[0], refrdir[1], refrdir[2]
				, hitPos[0] + nnhit[0] * 1e-2, hitPos[1] + nnhit[1] * 1e-2, hitPos[2] + nnhit[2] * 1e-2
				, light_x, light_y, light_z
				, hitlist
				, light_mode
				, x1, y1);
		}
		
		//float colMag = magnitude(diffCol);
		//diffCol[0] = diffCol[0] / colMag;
		//diffCol[1] = diffCol[1] / colMag;
		//diffCol[2] = diffCol[2] / colMag;

		//blendCol(col, factor
		//	, 255 * diffCol[0]
		//	, 255 * diffCol[1]
		//	, 255 * diffCol[2]);
		
		//if (light_mode >= 1) {
			/*
			setCol(col,
				255 * m * (min(1.0f, ((reflCol[0] * (fresneleffect)) + (refrCol[0] * (1 - fresneleffect)*transp)) * diffCol[0]))
				, 255 * m * (min(1.0f, ((reflCol[1] * (fresneleffect)) + (refrCol[1] * (1 - fresneleffect)*transp)) * diffCol[1]))
				, 255 * m * (min(1.0f, ((reflCol[2] * (fresneleffect)) + (refrCol[2] * (1 - fresneleffect)*transp)) * diffCol[2]))
			); 
			*/


		//}

	}
}

template<>
__device__ void RaytraceTris<MAX_RAY_DEPTH>(
	Texel* col
	, float factor
	, BVH* bvh
	, int numBVH
	, Vertex* verts
	, int numTris
	, float ray_x, float ray_y, float ray_z
	, float orig_x, float orig_y, float orig_z
	, float light_x, float light_y, float light_z
	, IntersectionResult* hitlist
	, int light_mode
	, float x1, float y1)
{

	float4 bgTex = tex2D(cudaTex3, (x1 + (WIDTH / 2)) / WIDTH, (y1 + (HEIGHT / 2)) / HEIGHT);

	//TODO maintain "russian roulette" probability factor to adjust energy here
	//Terminate the ray and add background color
	blendCol(col
		, factor
		, bgTex.x
		,  bgTex.y
		,  bgTex.z
	);
	
	//setCol(col
		//, 255, 255, 0);

	return;
}

#elif SHAPE_MODE==1

const int NUM_LIGHTS = 2;

template <int depth>
__device__ void Raytrace(Texel* col
	, float factor
	, Sphere* spheres
	, int numSpheres
	, float ray_x, float ray_y, float ray_z
	, float orig_x, float orig_y, float orig_z
	, int bouncedFromSphereIndex
	, float blendR, float blendG, float blendB)
{
	//set colours by screen position
	//setCol(col, ray_x, ray_y, ray_x / ray_y);
	//setCol(col, spheres[0].surfaceColor.r, spheres[0].surfaceColor.g, spheres[0].surfaceColor.b);
	//return;

	vec3 raydir = { 0 };

	raydir[0] = ray_x;
	raydir[1] = ray_y;
	raydir[2] = ray_z;

	float rdMag = magnitude(raydir);
	raydir[0] = ray_x / rdMag;
	raydir[1] = ray_y / rdMag;
	raydir[2] = ray_z / rdMag;

	vec3 rayorig = { 0 };

	rayorig[0] = orig_x;
	rayorig[1] = orig_y;
	rayorig[2] = orig_z;

	float tnear = INFINITY;

	Sphere* sphere = NULL;
	int selfIndex = -1;
	// find intersection of this ray with the sphere in the scene
	for (unsigned i = 0; i < numSpheres; ++i) {
		//if (i != bouncedFromSphereIndex) {
		float t = intersect(spheres[i], rayorig, raydir);
		if (t != -1.0f) {
			if (t < tnear) {
				tnear = t;
				sphere = &spheres[i];
				selfIndex = i;
			}
		}
		//}
	}

	//no intersection = background
	if (!sphere) {
		//set background only on primary rays

		if (depth == 1) setCol(col, 30, 30, 80);
		//NOTE: blend is slow!
		else {
			blendCol(col, factor, 30 * blendR, 30 * blendG, 80 * blendB);
		}
		return;
	}
	else if (sphere->emissionColor[0] > 0 || sphere->emissionColor[1] > 0 || sphere->emissionColor[2] > 0) {
		//light sources emit their colour 100%
		setCol(col, sphere->emissionColor[0] * 255, sphere->emissionColor[1] * 255, sphere->emissionColor[2] * 255);
		return;
	}

	//spheres are easy and fast because the normal is from centre-to-surface
	//and the hit point it the point on the closest intersection of radius and raydir
	vec3 phit = { 0 };
	phit[0] = orig_x + (raydir[0] * tnear);
	phit[1] = orig_y + (raydir[1] * tnear);
	phit[2] = orig_z + (raydir[2] * tnear);

	//teleport self-refracted ray

	if (selfIndex == bouncedFromSphereIndex) {

		Raytrace<depth + 1>(col
			, factor
			, spheres
			, numSpheres
			, ray_x, ray_y, ray_z
			, phit[0], phit[1], phit[2]
			, bouncedFromSphereIndex,
			blendR, blendG, blendB);
		return;
	}



	//Get normal at hit (and normalize it)
	vec3 nhit;
	nhit[0] = phit[0] - sphere->center[0];
	nhit[1] = phit[1] - sphere->center[1];
	nhit[2] = phit[2] - sphere->center[2];


	float mag = magnitude(nhit);
	vec3 nnhit = { 0 };
	nnhit[0] = nhit[0] / mag;
	nnhit[1] = nhit[1] / mag;
	nnhit[2] = nhit[2] / mag;

	bool inside = true;

	float rayhitdot = dot(raydir, nnhit);

	if (rayhitdot > 0) {
		inside = false;
	}

	nnhit[0] = ((inside) ? 1 : -1) * nnhit[0];
	nnhit[1] = ((inside) ? 1 : -1) * nnhit[1];
	nnhit[2] = ((inside) ? 1 : -1) * nnhit[2];

	float bias = sphere->radius*5e-3; // add some bias to the point from which we will be tracing

	//shadow query point is just outside sphere to avoid self-shadowing
	vec3 shadowQueryPoint = { 0 };
	shadowQueryPoint[0] = phit[0] + (nnhit[0] * bias);
	shadowQueryPoint[1] = phit[1] + (nnhit[1] * bias);
	shadowQueryPoint[2] = phit[2] + (nnhit[2] * bias);

	vec3 lighting = { 0 };
	float shadowMultiplier = 1.0f / NUM_LIGHTS;

	// it's a diffuse object, no need to raytrace any further
	for (unsigned i = 0; i < numSpheres; ++i) {
		//Light SHOULDN'T be the bouncedFrom sphere, but just in case
		if ((bouncedFromSphereIndex == -1 || i != bouncedFromSphereIndex)
			&&
			(spheres[i].emissionColor[0] > 0
				|| spheres[i].emissionColor[1] > 0
				|| spheres[i].emissionColor[2] > 0)) {

			float m = 1.0f;

			//do not allow direct self-lighting
			if (i != selfIndex) {
				// this is a light
				vec3 lightDirection = { 0 };
				lightDirection[0] = spheres[i].center[0] - phit[0];
				lightDirection[1] = spheres[i].center[1] - phit[1];
				lightDirection[2] = spheres[i].center[2] - phit[2];

				float lMag = magnitude(lightDirection);
				// normalize normal direction
				vec3 nld = { 0 };
				nld[0] = lightDirection[0] / lMag;
				nld[1] = lightDirection[1] / lMag;
				nld[2] = lightDirection[2] / lMag;

				for (unsigned j = 0; j < numSpheres; j++) {
					if (spheres[j].castShadows
						&& i != j && j != selfIndex) {
						if (hasIntersection(spheres[j], shadowQueryPoint, nld, lMag)) {

							shadowMultiplier = 0.01f;
							break;
						}
					}
				}

				//m does not change per pixel of object
				m = dot(nnhit, nld);
				if (sphere->transparency) {
					m = (m < 0) ? -m : max(0.0f, m);
				}

			}

			lighting[0] += (sphere->surfaceColor[0] * shadowMultiplier *
				m * spheres[i].emissionColor[0]);
			lighting[1] += (sphere->surfaceColor[1] * shadowMultiplier *
				m * spheres[i].emissionColor[1]);
			lighting[2] += (sphere->surfaceColor[2] * shadowMultiplier *
				m * spheres[i].emissionColor[2]);

			shadowMultiplier = 1.0f / NUM_LIGHTS;
		}

	}

	//diffuse colour at hit point
	blendCol(col, factor
		, blendR * (lighting[0]) * 255
		, blendG * (lighting[1]) * 255
		, blendB * (lighting[2]) * 255);

	//ray-traced transparency adds refraction colour
	//if the surface is lit
	if (//depth + 1 != MAX_RAY_DEPTH && 
		((lighting[0] + lighting[1] + lighting[2])*(blendR + blendG + blendB) > 0)
		&&
		(sphere->transparency || sphere->reflection)) {

		vec3 negraydir = { 0 };
		negraydir[0] = -raydir[0];
		negraydir[1] = -raydir[1];
		negraydir[2] = -raydir[2];

		//default: 50% reflection
		//other: transp+refl -> fresnel is computed
		float fresneleffect = 0.5f;

		float ior = 1.01f;
		float eta = (inside) ? ior : 1 / ior;

		vec3 nnnhit = { 0 };
		nnnhit[0] = -nnhit[0];
		nnnhit[1] = -nnhit[1];
		nnnhit[2] = -nnhit[2];

		if (sphere->transparency) {


			float facingratio = dot(nnhit, negraydir);
			fresneleffect = ((1 - facingratio) * (1 - facingratio)*(0.9f)) + 0.1;

			float cosi = dot(nnnhit, raydir);
			float k = 1 - (eta * eta * (1 - cosi * cosi));

			vec3 refrdir = { 0 };

			//TODO: negraydir is interesting but is it correct? was raydir

			refrdir[0] = (eta * negraydir[0]) + ((eta * (cosi - k)) * nnhit[0]);
			refrdir[1] = (eta * negraydir[1]) + ((eta * (cosi - k)) * nnhit[1]);
			refrdir[2] = (eta * negraydir[2]) + ((eta * (cosi - k)) * nnhit[2]);

			//normalise (and invert y)
			float refrMag = magnitude(refrdir);
			refrdir[0] = refrdir[0] / refrMag;
			refrdir[1] = refrdir[1] / refrMag;
			refrdir[2] = refrdir[2] / refrMag;

			vec3 refrOrig = { 0 };
			refrOrig[0] = phit[0] + (bias * nnnhit[0]);
			refrOrig[1] = phit[1] + (bias * nnnhit[1]);
			refrOrig[2] = phit[2] + (bias * nnnhit[2]);

			float refrFact = 0.5f*shadowMultiplier*sphere->transparency;
			if (refrFact > 0) {
				Raytrace<depth + 1>(col, refrFact, spheres, numSpheres
					, refrdir[0], refrdir[1], refrdir[2]
					, refrOrig[0], refrOrig[1], refrOrig[2]
					, selfIndex
					, blendR*lighting[0], blendG*lighting[1], blendB*lighting[2]);
			}
		}

		if (sphere->reflection) {
			vec3 reflOrig = { 0 };

			if (inside) {
				reflOrig[0] = phit[0] - (bias * nnhit[0]);
				reflOrig[1] = phit[1] - (bias * nnhit[1]);
				reflOrig[2] = phit[2] - (bias * nnhit[2]);
			}
			else {
				reflOrig[0] = phit[0] + (bias * nnhit[0]);
				reflOrig[1] = phit[1] + (bias * nnhit[1]);
				reflOrig[2] = phit[2] + (bias * nnhit[2]);
			}

			vec3 refldir = { 0 };
			refldir[0] = raydir[0] - (nnhit[0] * 2 * rayhitdot);
			refldir[1] = raydir[1] - (nnhit[1] * 2 * rayhitdot);
			refldir[2] = raydir[2] - (nnhit[2] * 2 * rayhitdot);

			float reflDirMag = magnitude(refldir);
			refldir[0] = refldir[0] / reflDirMag;
			refldir[1] = refldir[1] / reflDirMag;
			refldir[2] = refldir[2] / reflDirMag;

			//Fresnel-blend the reflection
			float reflFact = max(0.0f, shadowMultiplier*(0.5f - (0.5f*fresneleffect*sphere->reflection)));
			if (reflFact > 0) {
				Raytrace<depth + 1>(col, reflFact, spheres, numSpheres
					, refldir[0], refldir[1], refldir[2]
					, reflOrig[0], reflOrig[1], reflOrig[2]
					, selfIndex
					, blendR*lighting[0], blendG*lighting[1], blendB*lighting[2]);
			}
		}

	}
}


template <>
__device__ void Raytrace<MAX_RAY_DEPTH>(Texel* col
	, float factor
	, Sphere* spheres
	, int numSpheres
	, float ray_x, float ray_y, float ray_z
	, float orig_x, float orig_y, float orig_z
	, int bouncedFromSphereIndex
	, float blendR, float blendG, float blendB)
{

	//set pixel to red to visualise max_depth rays
	//setCol(col, 255, 0, 0);
	setCol(col, col->col[0]*factor, col->col[1]*factor, col->col[2]*factor);

	return;
}

#endif

/* Deal with verts in CUDA??
__global__ void update_vertex_data(Vertex* verts, int numTris) {


	unsigned int x_int = (blockIdx.x * blockDim.x + threadIdx.x) * 3;
	unsigned int y_int = (blockIdx.y * blockDim.y + threadIdx.y) * 3;

	if (x_int < numTris * 3 && ) {
		verts[]
	}

}
*/

/// <summary>
/// CUDA kernel function for ray-traced rendering into pixel array.
/// </summary>
/// <remarks>
/// Depend
/// </remarks>
///<param name = "pixels">A texel array to write rendered pixels into</param>
///<param name = "bvh">The BVH AABB array, used if BVH enabled</param>
///<param name = "numBVH">The number of BVH, used if BVH enabled</param>
///<param name = "verts">The vertices to ray-trace</param>
///<param name = "numTris">Number of triangles in the vertex array</param>
///<param name = "spheres">The spheres to ray-trace</param>
///<param name = "numSpheres">The number of spheres</param>
///<param name = "cam_x">The x-coordinate of the camera position</param>
///<param name = "cam_y">The y-coordinate of the camera position</param>
///<param name = "cam_z">The z-coordinate of the camera position</param>
///<param name = "light_x">The x-coordinate of the light position</param>
///<param name = "light_y">The y-coordinate of the light position</param>
///<param name = "light_z">The z-coordinate of the light position</param>
///<param name = "frame">The current sub-frame to render</param>
///<param name = "squareDim">The size of the deferred refresh square region</param>
///<param name = "factor">The influence of rendered pixels when added to pixels (1.0 will overwrite)</param>
///<param name = "light_mode">The interactive control for which rendering mode to use</param>
///<param name = "hitlist">A pre-allocated array to store hit results into</param>
__global__ void get_raytraced_pixels(
	Texel* pixels
	, BVH* bvh, int numBVH
	, Vertex* verts, int numTris
	, Sphere* spheres, int numSpheres
	, float cam_x, float cam_y, float cam_z
	, float light_x, float light_y, float light_z
	, int frame, int squareDim, float factor
	, int light_mode
	, IntersectionResult* hitlist) {

	unsigned int raw_x = (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned int raw_y = (blockIdx.y * blockDim.y + threadIdx.y);
	unsigned int x_int = raw_x * squareDim;
	unsigned int y_int = raw_y * squareDim;

	//frame patterns:
	/*
		frame	x   y
		1		0  0
		2		1  0
		3		0  1
		4		1  1
		----------------
		5	    2  0
		6		0  2
		7	    2  2
		8		1  2
		9	    2  1
		----------------
	   10		3  0
	   11       0  3
	   12       3  3
	   13       3  2
	   14       2  3
	   15       1  3
	   16       3  1
	*/

	//Use compile-time directive for less processing because this doesn't need to be switchable behaviour
#if DEFERRED_REFRESH_SQUARE_DIM==2
	if (frame == 2 || frame == 4) {
		x_int += 1;
	}
	if (frame == 3 || frame == 4) {
		y_int += 1;
	}

#elif DEFERRED_REFRESH_SQUARE_DIM==4

	if (frame == 2 || frame == 4 || frame == 8 || frame == 15) {
		x_int += 1;
	}
	else if (frame == 5 || frame == 7 || frame == 9 || frame == 14) {
		x_int += 2;
	}
	else if (frame == 10 || frame == 12 || frame == 13 || frame == 16) {
		x_int += 3;
	}

	if (frame == 3 || frame == 4 || frame == 9 || frame == 16) {
		y_int += 1;
	}
	else if (frame == 6 || frame == 7 || frame == 8 || frame == 13) {
		y_int += 2;
	}
	else if (frame == 11 || frame == 12 || frame == 14 || frame == 15) {
		y_int += 3;
	}
#endif


	int x = (x_int - (WIDTH / 2));
	int y = (y_int - (HEIGHT / 2));

	if (x_int < WIDTH && y_int < HEIGHT) {

#if SHAPE_MODE == 1		
		Raytrace<1>(
			//image
			&pixels[y_int * WIDTH + x_int]
			//factor to keep old color
			, factor
			//spheres
			, spheres, numSpheres
			//cam projection (smaller z = larger fov)
			, x, y, -600
			//cam pos
			, cam_x + (WIDTH / -2.0f), cam_y + (HEIGHT / -2.0f), cam_z + 1000,
			-1
			, 1.0f, 1.0f, 1.0f);

#elif SHAPE_MODE == 0
		//START PERSPECTIVE MATRIX
//TODO: calculate on host and copy to CUDA device once/when changed

	//aspect ratio: x/y

		const float zFar = -900;

		//float cam_x2 = (cam_x + (WIDTH / -2.0f)) + x;
		//float cam_y2 = (cam_y + (HEIGHT / -2.0f)) + y;
		/*
		Abandoned attempt to odraw light without ray-tracing it
		float lightDist = sqrtf((cam_x2 - light_x)*(cam_x2 - light_x) + (cam_y2 - light_y)*(cam_y2 - light_y));
		
		if (lightDist < 15.0f) {
			pixels[(y_int * WIDTH) + (x_int)].col[0] = 255;
			pixels[(y_int * WIDTH) + (x_int)].col[1] = 255;
			pixels[(y_int * WIDTH) + (x_int)].col[2] = 255;
		}
		else {
		*/
			RaytraceTris<1>(
				//image
				&pixels[(y_int * WIDTH) + (x_int)]
				//factor to keep old color
				, factor
				//bvh
				, bvh, numBVH
				//verts
				, verts, numTris
				//raydir (also acts as cam projection where smaller z = larger fov)
				, x, y, zFar
				//cam pos
				, cam_x, cam_y, cam_z
				, light_x, light_y, light_z
				, &hitlist[(raw_y * (WIDTH / (squareDim))) + (raw_x)]
				, light_mode
				, x, y
				);
		//}

#endif
	}

}

////END RAYTRACING

//HOST globals
void* cudaSpheres;
Sphere* spheres;
const int NUM_SPHERES = 6;

//const int NUM_TRIS = 5;

int NUM_TRIS = 0;
int NUM_BVH = 0;
int NUM_OCTREE = 0;

void* cudaVerts;
Vertex* vertData;

void* cudaBVH;
BVH* bvhData;

void* cudaHitList;
IntersectionResult* hitListData;

//Player State
bool spheresChanged;
bool vertsChanged;
float cam_x = 0.0f;
//-y is up
float cam_y = 0.0f;
float cam_z = 4000.0f;

//0 = diffuse, 1 = refl, 2 = refr, 3 refl+refr
int light_mode = 0;

float light_x = 0.0f;
float light_y = 0.0f;
float light_z = 1000.0f;

int framesRefreshRequired = 0;

float speed = 0.02f;

//render half the pixels in each dimension
//TODO: hardcoded for a trail size=2

//frameStep=0 forces full-frame rendering mode (disables sub-framing)
int frameStep = (DEFERRED_REFRESH_SQUARE_DIM == 1) ? 0 : 1;
//dimensions of refresh squares (square to get number of subframes)
int defferedSquareDim = DEFERRED_REFRESH_SQUARE_DIM;

bool keys[13] = { 0 };

std::chrono::system_clock::time_point WIN_CTIME = std::chrono::system_clock::now();
int oldTicks = 0;

//See:
//The callback function receives the keyboard key
//platform-specific scancode, key action and modifier bits.
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_RELEASE) {
		switch (key) {
		case (GLFW_KEY_R):
			keys[0] = false;
			break;
		case (GLFW_KEY_F):
			keys[1] = false;
			break;
		case(GLFW_KEY_A):
			keys[2] = false;
			break;
		case(GLFW_KEY_D):
			keys[3] = false;
			break;
		case (GLFW_KEY_W):
			keys[4] = false;
			break;
		case (GLFW_KEY_S):
			keys[5] = false;
			break;
		case (GLFW_KEY_E):
			keys[6] = false;
			break;
		case (GLFW_KEY_G):
			keys[7] = false;
			break;
		case (GLFW_KEY_Q):
			keys[8] = false;
			break;
		case (GLFW_KEY_Z):
			keys[9] = false;
			break;
		case (GLFW_KEY_X):
			keys[10] = false;
			break;
		case (GLFW_KEY_C):
			keys[11] = false;
			break;
		//toggle key is reset when handled
		//case (GLFW_KEY_TAB):
			//keys[12] = false;
			//break;
		default:
			break;
		}
	}
	else {
		switch (key) {
		case (GLFW_KEY_R):
			keys[0] = true;
			break;
		case (GLFW_KEY_F):
			keys[1] = true;
			break;
		case(GLFW_KEY_A):
			keys[2] = true;
			break;
		case(GLFW_KEY_D):
			keys[3] = true;
			break;
		case (GLFW_KEY_W):
			keys[4] = true;
			break;
		case (GLFW_KEY_S):
			keys[5] = true;
			break;
		case (GLFW_KEY_E):
			keys[6] = true;
			break;
		case (GLFW_KEY_G):
			keys[7] = true;
			break;
		case (GLFW_KEY_Q):
			keys[8] = true;
			break;
		case (GLFW_KEY_Z):
			keys[9] = true;
			break;
		case (GLFW_KEY_X):
			keys[10] = true;
			break;
		case (GLFW_KEY_C):
			keys[11] = true;
			break;
		case (GLFW_KEY_TAB):
			keys[12] = true;
			break;
		default:
			break;
		}
	}
}

class vulkanCudaApp {
public:
	void run() {
		initWindow();
		initVulkan();
		initCuda();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window;
	VkInstance instance;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	uint8_t vkDeviceUUID[VK_UUID_SIZE];
	VkDevice device;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkSurfaceKHR surface;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool descriptorPool;
	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkRenderPass renderPass;
	VkPipeline graphicsPipeline;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool;
	VkBuffer uniformBuffer;
	VkDeviceMemory uniformBufferMemory;
	std::vector<VkCommandBuffer> commandBuffers;
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore renderFinishedSemaphore;
	VkSemaphore cudaUpdateVkVertexBufSemaphore;
	VkSemaphore vkUpdateCudaVertexBufSemaphore;


	bool loadFromFile = true;
	
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;

	cudaArray *cudaTex1Array;
	cudaArray *cudaTex2Array;
	cudaArray *cudaTex3Array;

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;


	cudaEvent_t start, stop;

	stbi_uc* pixels;
	stbi_uc* texture1;
	stbi_uc* texture2;
	stbi_uc* texture3;

	size_t vertexBufSize = 0;
	bool startSubmit = 0;
	double AnimTime = 1.0f;


	VkDebugReportCallbackEXT callback;

#ifdef _WIN64
	PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
	PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
#else
	PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
	PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
#endif

	PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;

	// CUDA stuff
	cudaExternalMemory_t cudaExtMemPixelBuffer;
	cudaExternalSemaphore_t cudaExtCudaUpdateVkVertexBufSemaphore;
	cudaExternalSemaphore_t cudaExtVkUpdateCudaVertexBufSemaphore;
	void* cudaDevPixelptr = NULL;
	void* cudaDevIntersectResultsptr = NULL;
	cudaStream_t streamToRun;

	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL
		debugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType,
			uint64_t obj, size_t location, int32_t code,
			const char* layerPrefix, const char* msg, void* userData) {
		std::cerr << "validation layer: " << msg << std::endl;

		return VK_FALSE;
	}

	VkResult CreateDebugReportCallbackEXT(
		VkInstance instance,
		const VkDebugReportCallbackCreateInfoEXT* pCreateInfo,
		const VkAllocationCallbacks* pAllocator,
		VkDebugReportCallbackEXT* pCallback) {
		auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(
			instance, "vkCreateDebugReportCallbackEXT");
		if (func != nullptr) {
			return func(instance, pCreateInfo, pAllocator, pCallback);
		}
		else {
			return VK_ERROR_EXTENSION_NOT_PRESENT;
		}
	}

	void DestroyDebugReportCallbackEXT(VkInstance instance,
		VkDebugReportCallbackEXT callback,
		const VkAllocationCallbacks* pAllocator) {
		auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(
			instance, "vkDestroyDebugReportCallbackEXT");
		if (func != nullptr) {
			func(instance, callback, pAllocator);
		}
	}

	void setupDebugCallback() {
		if (enableValidationLayers) {

			VkDebugReportCallbackCreateInfoEXT createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
			createInfo.flags =
				VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
			createInfo.pfnCallback = debugCallback;

			if (CreateDebugReportCallbackEXT(instance, &createInfo, nullptr,
				&callback) != VK_SUCCESS) {
				throw std::runtime_error("failed to set up debug callback!");
			}
		}
	}

	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan-CUDA Interop Sinewave",
			nullptr, nullptr);
	}

	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error(
				"validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Vulkan CUDA Sinewave";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> enabledExtensionNameList;
		enabledExtensionNameList.push_back(
			VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		enabledExtensionNameList.push_back(
			VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
		enabledExtensionNameList.push_back(
			VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

		for (unsigned int i = 0; i < glfwExtensionCount; i++) {
			enabledExtensionNameList.push_back(glfwExtensions[i]);
		}
		if (enableValidationLayers) {
			enabledExtensionNameList.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
			createInfo.enabledLayerCount =
				static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensionNameList.size());
		createInfo.ppEnabledExtensionNames = enabledExtensionNameList.data();

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
		else {
			std::cout << "Instance created successfully!!\n";
		}

		fpGetPhysicalDeviceProperties2 =
			(PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
				instance, "vkGetPhysicalDeviceProperties2");
		if (fpGetPhysicalDeviceProperties2 == NULL) {
			throw std::runtime_error(
				"Vulkan: Proc address for \"vkGetPhysicalDeviceProperties2KHR\" not "
				"found.\n");
		}

#ifdef _WIN64
		fpGetMemoryWin32HandleKHR =
			(PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddr(
				instance, "vkGetMemoryWin32HandleKHR");
		if (fpGetMemoryWin32HandleKHR == NULL) {
			throw std::runtime_error(
				"Vulkan: Proc address for \"vkGetMemoryWin32HandleKHR\" not "
				"found.\n");
		}
#else
		fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(
			instance, "vkGetMemoryFdKHR");
		if (fpGetMemoryFdKHR == NULL) {
			throw std::runtime_error(
				"Vulkan: Proc address for \"vkGetMemoryFdKHR\" not found.\n");
		}
#endif
	}

	void initVulkan() {
		createInstance();
		setupDebugCallback();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		getKhrExtensionsFn();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createDescriptorSetLayout();
		//createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();

		//Added textures
		createTextureImages();

		createUniformBuffer();
		createDescriptorPool();
		createDescriptorSet();
		createCommandBuffers();
		createSyncObjects();
		createSyncObjectsExt();
	}

	void initCuda() {
		setCudaVkDevice();
		cudaVkImportVertexMem();
		cudaInitVertexMem();
		cudaVkImportSemaphore();
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;

		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		else {
			std::cout << "Found devices = " << deviceCount << std::endl;
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}
		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}

		std::cout << "Selected physical device = " << physicalDevice << std::endl;

		VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
		vkPhysicalDeviceIDProperties.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
		vkPhysicalDeviceIDProperties.pNext = NULL;

		VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
		vkPhysicalDeviceProperties2.sType =
			VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

		fpGetPhysicalDeviceProperties2(physicalDevice,
			&vkPhysicalDeviceProperties2);

		memcpy(vkDeviceUUID, vkPhysicalDeviceIDProperties.deviceUUID,
			sizeof(vkDeviceUUID));
	}

	int setCudaVkDevice() {
		int current_device = 0;
		int device_count = 0;
		int devices_prohibited = 0;

		cudaDeviceProp deviceProp;
		checkCudaErrors(cudaGetDeviceCount(&device_count));

		if (device_count == 0) {
			fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
			exit(EXIT_FAILURE);
		}

		// Find the GPU which is selected by Vulkan
		while (current_device < device_count) {
			cudaGetDeviceProperties(&deviceProp, current_device);

			if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
				// Compare the cuda device UUID with vulkan UUID
				int ret = memcmp(&deviceProp.uuid, &vkDeviceUUID, VK_UUID_SIZE);
				if (ret == 0) {
					checkCudaErrors(cudaSetDevice(current_device));
					checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
					printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
						current_device, deviceProp.name, deviceProp.major,
						deviceProp.minor);

					return current_device;
				}

			}
			else {
				devices_prohibited++;
			}

			current_device++;
		}

		if (devices_prohibited == device_count) {
			fprintf(stderr,
				"CUDA error:"
				" No Vulkan-CUDA Interop capable GPU found.\n");
			exit(EXIT_FAILURE);
		}

		return -1;
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);
		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() &&
				!swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
			nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
			availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(),
			deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
			nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
			queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueCount > 0 &&
				queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (queueFamily.queueCount > 0 && presentSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}
			i++;
		}
		return indices;
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
			&details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
			nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
				details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
			&presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(
				device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(
		const std::vector<VkSurfaceFormatKHR>& availableFormats) {

		//If no query results or preferred surface format is not available, use whatever is available
		//Otherwise, we can use "preferred"

		VkFormat preferredFormat = VK_FORMAT_B8G8R8A8_UNORM;
		VkColorSpaceKHR preferredColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

		if (availableFormats.size() == 1 &&
			availableFormats[0].format == VK_FORMAT_UNDEFINED) {
			return { preferredFormat, preferredColorSpace };
		}

		std::cout << "Available surface capabilities:" << std::endl;
		std::cout << "Color space" << ":" << "Format" << std::endl;
		for (const auto& availableFormat : availableFormats) {

			std::cout << availableFormat.colorSpace << ":" << availableFormat.format << std::endl;

			if (availableFormat.format == preferredFormat
				&& availableFormat.colorSpace == preferredColorSpace) {
				return availableFormat;
			}
		}
		std::cout << "Preferred surface format not supported. Using available." << std::endl;
		std::cout << availableFormats[0].colorSpace << ":" << availableFormats[0].format << std::endl;
		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(
		const std::vector<VkPresentModeKHR> availablePresentModes) {
		VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
			else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
				bestMode = availablePresentMode;
			}
		}

		return bestMode;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width !=
			std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			VkExtent2D actualExtent = { WIDTH, HEIGHT };

			actualExtent.width = std::max(
				capabilities.minImageExtent.width,
				std::min(capabilities.maxImageExtent.width, actualExtent.width));
			actualExtent.height = std::max(
				capabilities.minImageExtent.height,
				std::min(capabilities.maxImageExtent.height, actualExtent.height));

			return actualExtent;
		}
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<int> uniqueQueueFamilies = { indices.graphicsFamily,
											 indices.presentFamily };

		float queuePriority = 1.0f;
		for (int queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};

		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

		createInfo.pEnabledFeatures = &deviceFeatures;
		std::vector<const char*> enabledExtensionNameList;

		for (int i = 0; i < deviceExtensions.size(); i++) {
			enabledExtensionNameList.push_back(deviceExtensions[i]);
		}
		if (enableValidationLayers) {
			createInfo.enabledLayerCount =
				static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}
		createInfo.enabledExtensionCount =
			static_cast<uint32_t>(enabledExtensionNameList.size());
		createInfo.ppEnabledExtensionNames = enabledExtensionNameList.data();

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}
		vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily, 0, &presentQueue);
	}

	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport =
			querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat =
			chooseSwapSurfaceFormat(swapChainSupport.formats);

		VkPresentModeKHR presentMode =
			chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 &&
			imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}


		//For info on swapchain creation:
		//https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain

		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

		//// my crap
		//createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		//createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;


		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { (uint32_t)indices.graphicsFamily,
										 (uint32_t)indices.presentFamily };

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;      // Optional
			createInfo.pQueueFamilyIndices = nullptr;  // Optional
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}
		else {
			std::cout << "Swapchain created.\n";
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
			swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;

			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			if (vkCreateImageView(device, &createInfo, nullptr,
				&swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}
		}
	}

	void createDescriptorSetLayout() {
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		uboLayoutBinding.pImmutableSamplers = nullptr;  // Optional

		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 1;
		layoutInfo.pBindings = &uboLayoutBinding;

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
			&descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("shader_sine.vert");
		auto fragShaderCode = readFile("shader_sine.frag");

		VkShaderModule vertShaderModule;
		VkShaderModule fragShaderModule;

		vertShaderModule = createShaderModule(vertShaderCode);
		fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType =
			VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType =
			VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo,
														  fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType =
			VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;


		//TODO: not
		VkVertexInputBindingDescription bindingDescription; //= Vertex::getBindingDescription();
		VkVertexInputAttributeDescription attributeDescriptions; //= Vertex::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = 1;
			//static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = &attributeDescriptions;// .data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType =
			VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType =
			VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;  // Optional
		rasterizer.depthBiasClamp = 0.0f;           // Optional
		rasterizer.depthBiasSlopeFactor = 0.0f;     // Optional

		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType =
			VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;           // Optional
		multisampling.pSampleMask = nullptr;             // Optional
		multisampling.alphaToCoverageEnable = VK_FALSE;  // Optional
		multisampling.alphaToOneEnable = VK_FALSE;       // Optional

		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
		colorBlendAttachment.dstColorBlendFactor =
			VK_BLEND_FACTOR_ZERO;                                        // Optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;             // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
		colorBlendAttachment.dstAlphaBlendFactor =
			VK_BLEND_FACTOR_ZERO;                             // Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;  // Optional

		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType =
			VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;  // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;  // Optional
		colorBlending.blendConstants[1] = 0.0f;  // Optional
		colorBlending.blendConstants[2] = 0.0f;  // Optional
		colorBlending.blendConstants[3] = 0.0f;  // Optional

#if 0
		VkDynamicState dynamicStates[] = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_LINE_WIDTH
		};

		VkPipelineDynamicStateCreateInfo dynamicState = {};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = 2;
		dynamicState.pDynamicStates = dynamicStates;
#endif
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;                  // Optional
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;  // Optional
		pipelineLayoutInfo.pushConstantRangeCount = 0;          // Optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr;       // Optional

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
			&pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = nullptr;  // Optional
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = nullptr;  // Optional
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;  // Optional
		pipelineInfo.basePipelineIndex = -1;               // Optional

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
			nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}
		else {
			std::cout << "Pipeline created successfully!!\n";
		}
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	void createRenderPass() {
		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

		//pColorAttachments lists which of the render passs attachments will be used as color attachments 
		//in the subpass, and what layout each attachment will be in during the subpass. 
		//Each element of the array corresponds to a fragment shader output location, 
		//i.e. if the shader declared an output variable layout(location=X) then it uses the 
		//attachment provided in pColorAttachments[X].
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;

		//VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT specifies the stage of the pipeline 
		//after blending where the final color values are output from the pipeline. 
		//This stage also includes subpass load and store operations and multisample 
		//resolve operations for framebuffer attachments with a color or depth/stencil format.
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
			VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			VkImageView attachments[] = { swapChainImageViews[i] };

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr,
				&swapChainFramebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;
		poolInfo.flags = 0;  // Optional

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
		VkMemoryPropertyFlags properties, VkBuffer& buffer,
		VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex =
			findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	void createBufferExtMem(VkDeviceSize size, VkBufferUsageFlags usage,
		VkMemoryPropertyFlags properties,
		VkExternalMemoryHandleTypeFlagsKHR extMemHandleType,
		VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

#ifdef _WIN64
		WindowsSecurityAttributes winSecurityAttributes;

		VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR = {};
		vulkanExportMemoryWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
		vulkanExportMemoryWin32HandleInfoKHR.pNext = NULL;
		vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
		vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
			DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
		vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif
		VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
		vulkanExportMemoryAllocateInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
		vulkanExportMemoryAllocateInfoKHR.pNext =
			extMemHandleType & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
			? &vulkanExportMemoryWin32HandleInfoKHR
			: NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes = extMemHandleType;
#else
		vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex =
			findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to allocate external buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}


	void cudaInitVertexMem() {
		checkCudaErrors(cudaStreamCreate(&streamToRun));

		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

#if SHAPE_MODE == 1
		vec3 c, sc, ec;
		c[0] = -400.0f; c[1] = 0.0f; c[2] = -6500.0f;

		sc[0] = 1.0f; sc[1] = 0.1f; sc[2] = 0.1f;
		ec[0] = 0; ec[1] = 0; ec[2] = 0;

		spheres = (Sphere*)malloc(NUM_SPHERES * sizeof(Sphere));

		//TODO: try 5 spheres in a line horizontally
		//TODO: refraction should be flipped

		spheres[0] = Sphere(c
			, 500.0f
			, sc
			, 0
			, 0
			, ec
			, true);

		//c[0] = 0.0f; c[1] = 0.0f; c[2] = -3000.0f;
		c[0] = -1000.0f; c[1] = 0.0f; c[2] = -5000.0f;
		ec[0] = 0.6f; ec[1] = 1.0f; ec[2] = 0.6f;

		spheres[1] = Sphere(c
			, 20.0f
			, ec
			, 0
			, 0
			, ec
			, false);

		c[0] = 0.0f; c[1] = 0.0f; c[2] = -250.0f;
		ec[0] = 1; ec[1] = 1; ec[2] = 1;

		spheres[2] = Sphere(c
			, 16.0f
			, ec
			, 0
			, 0
			, ec
			, false);

		//c[0] = 0.0f; c[1] = 9000000.0f; c[2] = -1000.0f;
		//sc[0] = 1.0f; sc[1] = 1.0f; sc[2] = 1.0f;
		//c[0] = 0.0f; c[1] = 0.0f; c[2] = -20.0f;
		c[0] = 800.0f; c[1] = 0.0f; c[2] = -5000.0f;
		sc[0] = 0.1f; sc[1] = 0.1f; sc[2] = 1.0f;
		ec[0] = 0; ec[1] = 0; ec[2] = 0;

		spheres[3] = Sphere(c
			, 200.0f
			, sc
			, 0.1f
			, 0
			, ec
			, true);

		c[0] = -2500.0f; c[1] = 0.0f; c[2] = -5000.0f;
		sc[0] = 0.5f; sc[1] = 0.6f; sc[2] = 1.0f;
		ec[0] = 0; ec[1] = 0; ec[2] = 0;

		spheres[4] = Sphere(c
			, 400.0f
			, sc
			, 1
			, 1
			, ec
			, true);


		c[0] = 0.0f; c[1] = 9000000.0f; c[2] = -1000.0f;
		sc[0] = 0.1f; sc[1] = 0.1f; sc[2] = 0.1f;
		ec[0] = 0; ec[1] = 0; ec[2] = 0;

		spheres[5] = Sphere(c
			, 10000000.0f
			, sc
			, 0
			, 0
			, ec
			, false);

		checkCudaErrors(cudaMallocManaged((void**)&cudaSpheres, NUM_SPHERES * sizeof(Sphere), cudaMemAttachGlobal));
		checkCudaErrors(cudaMemcpy(cudaSpheres, spheres, NUM_SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice));

#elif SHAPE_MODE == 0

		/*
		vertData = (Vertex*)malloc(3 * NUM_TRIS * sizeof(Vertex));

		// BEGIN TRI 1

		vertData[0] = Vertex();

		vertData[0].pos[0] = 400;
		vertData[0].pos[1] = 400;
		vertData[0].pos[2] = -100.0f;
		vertData[0].pos[3] = 1.0f;

		vertData[0].color[0] = 0.0f;
		vertData[0].color[1] = 0.0f;
		vertData[0].color[2] = 1.0f;

		vertData[1] = Vertex();

		vertData[1].pos[0] = -512.0f;
		vertData[1].pos[1] = -512.0f;
		vertData[1].pos[2] = -100.0f;
		vertData[1].pos[3] = 1.0f;

		vertData[1].color[0] = 0.0f;
		vertData[1].color[1] = 1.0f;
		vertData[1].color[2] = 0.0f;

		vertData[2] = Vertex();

		vertData[2].pos[0] = 512.0f;
		vertData[2].pos[1] = -512.0f;
		vertData[2].pos[2] = -100.0f;
		vertData[2].pos[3] = 1.0f;

		vertData[2].color[0] = 1.0f;
		vertData[2].color[1] = 0.0f;
		vertData[2].color[2] = 0.0f;


		//END TRI 1

		// BEGIN TRI 2

		vertData[3] = Vertex();

		vertData[3].pos[0] = 512;
		vertData[3].pos[1] = -512;
		vertData[3].pos[2] = -100.0f;
		vertData[3].pos[3] = 1.0f;

		vertData[3].color[0] = 1.0f;
		vertData[3].color[1] = 0.0f;
		vertData[3].color[2] = 0.0f;

		vertData[4] = Vertex();

		vertData[4].pos[0] = -512.0f;
		vertData[4].pos[1] = -512.0f;
		vertData[4].pos[2] = -100.0f;
		vertData[4].pos[3] = 1.0f;

		vertData[4].color[0] = 0.0f;
		vertData[4].color[1] = 1.0f;
		vertData[4].color[2] = 0.0f;

		vertData[5] = Vertex();

		vertData[5].pos[0] = -512.0f;
		vertData[5].pos[1] = -512.0f;
		vertData[5].pos[2] = -300.0f;
		vertData[5].pos[3] = 1.0f;

		vertData[5].color[0] = 0.0f;
		vertData[5].color[1] = 0.0f;
		vertData[5].color[2] = 1.0f;


		//END TRI 2

		// BEGIN TRI 3

		vertData[6] = Vertex();

		vertData[6].pos[0] = -512;
		vertData[6].pos[1] = 512;
		vertData[6].pos[2] = -100.0f;
		vertData[6].pos[3] = 1.0f;

		vertData[6].color[0] = 0.0f;
		vertData[6].color[1] = 1.0f;
		vertData[6].color[2] = 0.0f;

		vertData[7] = Vertex();

		vertData[7].pos[0] = -512.0f;
		vertData[7].pos[1] = -512.0f;
		vertData[7].pos[2] = -100.0f;
		vertData[7].pos[3] = 1.0f;

		vertData[7].color[0] = 1.0f;
		vertData[7].color[1] = 0.0f;
		vertData[7].color[2] = 0.0f;

		vertData[8] = Vertex();

		vertData[8].pos[0] = 512.0f;
		vertData[8].pos[1] = 512.0f;
		vertData[8].pos[2] = -100.0f;
		vertData[8].pos[3] = 1.0f;

		vertData[8].color[0] = 0.0f;
		vertData[8].color[1] = 0.0f;
		vertData[8].color[2] = 1.0f;


		//END TRI 3

		// BEGIN TRI 4

		vertData[9] = Vertex();

		vertData[9].pos[0] = -3000;
		vertData[9].pos[1] = 1500;
		vertData[9].pos[2] = -100.0f;
		vertData[9].pos[3] = 1.0f;

		vertData[9].color[0] = 1.0f;
		vertData[9].color[1] = 1.0f;
		vertData[9].color[2] = 1.0f;

		vertData[10] = Vertex();

		vertData[10].pos[0] = 3000;
		vertData[10].pos[1] = 1500;
		vertData[10].pos[2] = -100.0f;
		vertData[10].pos[3] = 1.0f;

		vertData[10].color[0] = 1.0f;
		vertData[10].color[1] = 1.0f;
		vertData[10].color[2] = 1.0f;

		vertData[11] = Vertex();

		vertData[11].pos[0] = 1500;
		vertData[11].pos[1] = 1500;
		vertData[11].pos[2] = -3000;
		vertData[11].pos[3] = 1.0f;

		vertData[11].color[0] = 1.0f;
		vertData[11].color[1] = 1.0f;
		vertData[11].color[2] = 1.0f;

		//END TRI 4

		// BEGIN TRI 5

		vertData[12] = Vertex();

		vertData[12].pos[0] = -100;
		vertData[12].pos[1] = -2050;
		vertData[12].pos[2] = -100;
		vertData[12].pos[3] = 1.0f;

		vertData[12].color[0] = 1.0f;
		vertData[12].color[1] = 1.0f;
		vertData[12].color[2] = 1.0f;

		vertData[13] = Vertex();

		vertData[13].pos[0] = 0;
		vertData[13].pos[1] = -2100;
		vertData[13].pos[2] = -100;
		vertData[13].pos[3] = 1.0f;

		vertData[13].color[0] = 1.0f;
		vertData[13].color[1] = 1.0f;
		vertData[13].color[2] = 1.0f;

		vertData[14] = Vertex();

		vertData[14].pos[0] = 0;
		vertData[14].pos[1] = -2000;
		vertData[14].pos[2] = -100;
		vertData[14].pos[3] = 1.0f;

		vertData[14].color[0] = 1.0f;
		vertData[14].color[1] = 1.0f;
		vertData[14].color[2] = 1.0f;

		//END TRI 5
		*/

		int numVerts = OBJLoader::loadRawVertexList("Assets/cubey.obj", &vertData);
		
		vertData = (Vertex*)malloc(numVerts * sizeof(Vertex));
		OBJLoader::loadVertices(vertData, numVerts);

		int numBVH = OBJLoader::countBVHNeeded(vertData, numVerts);
		bvhData = (BVH*)malloc(numBVH * sizeof(BVH));
		int numOctree = OBJLoader::createBVH(bvhData, numBVH, vertData, numVerts);

		NUM_TRIS = numVerts / 3;
		NUM_BVH = numBVH;
		NUM_OCTREE = numOctree;

		//Transfer vert data to GPU
		checkCudaErrors(cudaMallocManaged((void**)&cudaVerts, numVerts * sizeof(Vertex), cudaMemAttachGlobal));
		checkCudaErrors(cudaMemcpy(cudaVerts, vertData, numVerts * sizeof(Vertex), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMallocManaged((void**)&cudaBVH, numBVH * sizeof(BVH), cudaMemAttachGlobal));
		checkCudaErrors(cudaMemcpy(cudaBVH, bvhData, numBVH * sizeof(BVH), cudaMemcpyHostToDevice));


		//Initialise memory required to store intersection results
		size_t hitlistSize = (WIDTH * HEIGHT * sizeof(IntersectionResult)) / ((DEFERRED_REFRESH_SQUARE_DIM)*(DEFERRED_REFRESH_SQUARE_DIM));
		//size_t hitlistSize = (WIDTH * HEIGHT * sizeof(IntersectionResult));

		hitListData = (IntersectionResult*)malloc(hitlistSize);

		//Allocate memory for intersection result list
		checkCudaErrors(cudaMallocManaged(
			(void**)&cudaHitList
			, hitlistSize
			, cudaMemAttachGlobal));

		checkCudaErrors(cudaMemcpy(
			cudaHitList
			, hitListData
			//Only need as many intersection results as will be used in each CUDA call (e.g. deferred refresh sub-frames)
			, hitlistSize
			, cudaMemcpyHostToDevice));
			
		
#endif
		//COLOR TEXTURE
		// Copy to device memory some data located at address h_data in host memory
		cudaChannelFormatDesc channelDesc = //cudaCreateChannelDesc(sizeof(float), sizeof(float), sizeof(float), sizeof(float), cudaChannelFormatKindFloat);
			cudaCreateChannelDesc<uchar4>();

		checkCudaErrors(cudaMallocArray(&cudaTex1Array, &channelDesc, WIDTH, HEIGHT));
		cudaMemcpyToArray(cudaTex1Array, 0, 0, texture1, WIDTH*HEIGHT * sizeof(uchar4), cudaMemcpyHostToDevice);

		// Set texture parameters
		cudaTex1.normalized = true;
		cudaTex1.filterMode = cudaFilterModeLinear; //= cudaFilterModePoint;  
		cudaTex1.addressMode[0] = cudaAddressModeWrap;
		cudaTex1.addressMode[1] = cudaAddressModeWrap;
		cudaTex1.addressMode[2] = cudaAddressModeWrap;

		// Bind the array to the texture 
		checkCudaErrors(cudaBindTextureToArray(cudaTex1, cudaTex1Array, channelDesc));

		//BUMP TEXTURE
		checkCudaErrors(cudaMallocArray(&cudaTex2Array, &channelDesc, WIDTH, HEIGHT));
		cudaMemcpyToArray(cudaTex2Array, 0, 0, texture2, WIDTH*HEIGHT * sizeof(uchar4), cudaMemcpyHostToDevice);

		// Set texture parameters
		cudaTex2.normalized = true;
		cudaTex2.filterMode = cudaFilterModeLinear; //= cudaFilterModePoint;  
		cudaTex2.addressMode[0] = cudaAddressModeWrap;
		cudaTex2.addressMode[1] = cudaAddressModeWrap;
		cudaTex2.addressMode[2] = cudaAddressModeWrap;

		// Bind the array to the texture
		checkCudaErrors(cudaBindTextureToArray(cudaTex2, cudaTex2Array, channelDesc));

		//BG TEXTURE
		checkCudaErrors(cudaMallocArray(&cudaTex3Array, &channelDesc, WIDTH, HEIGHT));
		cudaMemcpyToArray(cudaTex3Array, 0, 0, texture3, WIDTH*HEIGHT * sizeof(uchar4), cudaMemcpyHostToDevice);

		// Set texture parameters
		cudaTex3.normalized = true;
		cudaTex3.filterMode = cudaFilterModeLinear; //= cudaFilterModePoint;  
		cudaTex3.addressMode[0] = cudaAddressModeWrap;
		cudaTex3.addressMode[1] = cudaAddressModeWrap;
		cudaTex3.addressMode[2] = cudaAddressModeWrap;

		// Bind the array to the texture
		checkCudaErrors(cudaBindTextureToArray(cudaTex3, cudaTex3Array, channelDesc));


		checkCudaErrors(cudaStreamSynchronize(streamToRun));
	}

	void createUniformBuffer() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);
		createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			uniformBuffer, uniformBufferMemory);
	}

	uint32_t findMemoryType(uint32_t typeFilter,
		VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags &
				properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void getKhrExtensionsFn() {
#ifdef _WIN64

		fpGetSemaphoreWin32HandleKHR =
			(PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
				device, "vkGetSemaphoreWin32HandleKHR");
		if (fpGetSemaphoreWin32HandleKHR == NULL) {
			throw std::runtime_error(
				"Vulkan: Proc address for \"vkGetSemaphoreWin32HandleKHR\" not "
				"found.\n");
		}
#else
		fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
			device, "vkGetSemaphoreFdKHR");
		if (fpGetSemaphoreFdKHR == NULL) {
			throw std::runtime_error(
				"Vulkan: Proc address for \"vkGetSemaphoreFdKHR\" not found.\n");
		}
#endif
	}

	void createCommandBuffers() {
		commandBuffers.resize(swapChainFramebuffers.size());

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}

		for (size_t i = 0; i < commandBuffers.size(); i++) {
			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
			beginInfo.pInheritanceInfo = nullptr;  // Optional

			if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			VkRenderPassBeginInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[i];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swapChainExtent;

			VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues = &clearColor;

			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo,
				VK_SUBPASS_CONTENTS_INLINE);

			doBlit(commandBuffers[i], swapChainImages[i]);

			vkCmdEndRenderPass(commandBuffers[i]);

			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}


	void doBlit(VkCommandBuffer commandBuffer, VkImage dstImage) {


		// Define the region to blit (we will blit the whole swapchain image)
		/*
		VkOffset3D blitSize;
		blitSize.x = WIDTH;
		blitSize.y = HEIGHT;
		blitSize.z = 1;
		VkImageBlit imageBlitRegion{};
		imageBlitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageBlitRegion.srcSubresource.layerCount = 1;
		imageBlitRegion.srcOffsets[1] = blitSize;
		imageBlitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageBlitRegion.dstSubresource.layerCount = 1;
		imageBlitRegion.dstOffsets[1] = blitSize;
		*/

		//TODO: writing direct to stagingBuffer in drawFrame is not working
		//Perhaps export pixels to CUDA and write to that and use below to convert to vkimage?

		//void* data;
		//vkMapMemory(device, stagingBufferMemory, 0, WIDTH*HEIGHT*4*sizeof(stbi_uc), 0, &data);
		//memcpy(data, pixels, static_cast<size_t>(WIDTH*HEIGHT * 4 * sizeof(stbi_uc)));
		//vkUnmapMemory(device, stagingBufferMemory);

		transitionImageLayout(commandBuffer, dstImage, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(commandBuffer, stagingBuffer, dstImage, static_cast<uint32_t>(WIDTH), static_cast<uint32_t>(HEIGHT));
		//copyBufferToImage(commandBuffer, stagingBuffer, dstImage, static_cast<uint32_t>(WIDTH), static_cast<uint32_t>(HEIGHT));

		transitionImageLayout(commandBuffer, dstImage, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		// Issue the blit command

		/*
		vkCmdBlitImage(
			commandBuffer,
			textureImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&imageBlitRegion,
			VK_FILTER_NEAREST);
		   */

		   //WHAT THE FUCK IS HAPPENING
		   /*
		   VkImageCopy* pRegions = new VkImageCopy();
		   //VkImageBlit* pRegions = new VkImageBlit();
		   //pRegions->dstOffsets[0] = VkOffset3D{ 0, 0, 0 };
		   //pRegions->dstOffsets[1] = VkOffset3D{ 100, 100, 0 };
		   pRegions->dstOffset = VkOffset3D{ 100, 100, 0 };
		   pRegions->dstSubresource = VkImageSubresourceLayers();
		   pRegions->dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		   pRegions->dstSubresource.baseArrayLayer = 0;
		   pRegions->dstSubresource.layerCount = 1;
		   pRegions->dstSubresource.mipLevel = 0;

		   //pRegions->srcOffsets[0] = VkOffset3D{ 0, 0, 0 };
		   //pRegions->srcOffsets[1] = VkOffset3D{ 100, 100, 0 };
		   pRegions->srcOffset = VkOffset3D{ 100, 100, 0 };
		   pRegions->srcSubresource = VkImageSubresourceLayers();
		   pRegions->srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		   pRegions->srcSubresource.baseArrayLayer = 0;
		   pRegions->srcSubresource.layerCount = 1;
		   pRegions->srcSubresource.mipLevel = 0;



		   vkCmdCopyImage(
			   commandBuffer,
			   textureImage,
			   VK_IMAGE_LAYOUT_GENERAL,
			   dstImage,
			   VK_IMAGE_LAYOUT_GENERAL,
			   1,
			   pRegions);
			   */

			   /*
			   vkCmdBlitImage(
				   commandBuffer,
				   textureImage,
				   VK_IMAGE_LAYOUT_GENERAL,
				   dstImage,
				   VK_IMAGE_LAYOUT_GENERAL,
				   1,
				   pRegions,
				   VK_FILTER_LINEAR);
				 */

	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}

	static std::vector<char> readFile(const std::string& filename) {
		char* file_path = sdkFindFilePath(filename.c_str(), execution_path.c_str());

		std::ifstream file(file_path, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open shader spv file!\n");
		}
		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);
		file.seekg(0);
		file.read(buffer.data(), fileSize);
		file.close();

		return buffer;
	}



	void mainLoop() {
		//updateUniformBuffer();
		glfwSetKeyCallback(window, key_callback);

		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(device);
	}

/*
void updateUniformBuffer() {
		UniformBufferObject ubo = {};

		mat4x4_identity(ubo.model);
		mat4x4 Model;
		mat4x4_dup(Model, ubo.model);
		mat4x4_rotate(ubo.model, Model, 1.0f, 0.0f, 1.0f, degreesToRadians(5.0f));

		vec3 eye = { 2.0f, 2.0f, 2.0f };
		vec3 center = { 0.0f, 0.0f, 0.0f };
		vec3 up = { 0.0f, 0.0f, 1.0f };
		mat4x4_look_at(ubo.view, eye, center, up);
		mat4x4_perspective(ubo.proj, degreesToRadians(45.0f),
			swapChainExtent.width / (float)swapChainExtent.height,
			0.1f, 10.0f);
		ubo.proj[1][1] *= -1;
		void* data;
		vkMapMemory(device, uniformBufferMemory, 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniformBufferMemory);
	}
	*/

	void createDescriptorPool() {
		VkDescriptorPoolSize poolSize = {};
		poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSize.descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 1;
		poolInfo.pPoolSizes = &poolSize;
		poolInfo.maxSets = 1;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createDescriptorSet() {
		VkDescriptorSetLayout layouts[] = { descriptorSetLayout };
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = layouts;

		if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor set!");
		}

		VkDescriptorBufferInfo bufferInfo = {};
		bufferInfo.buffer = uniformBuffer;
		bufferInfo.offset = 0;
		bufferInfo.range = sizeof(UniformBufferObject);

		VkWriteDescriptorSet descriptorWrite = {};
		descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet = descriptorSet;
		descriptorWrite.dstBinding = 0;
		descriptorWrite.dstArrayElement = 0;
		descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pBufferInfo = &bufferInfo;
		descriptorWrite.pImageInfo = nullptr;        // Optional
		descriptorWrite.pTexelBufferView = nullptr;  // Optional

		vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
	}

	void drawFrame() {


		if (keys[0] || keys[1]) {
#if SHAPE_MODE == 1
			spheres[2].center[1] += (keys[0]) ? -speed : speed;
			spheresChanged = true;
#elif SHAPE_MODE == 0
			//vertsChanged = true;
			//vertData[((NUM_TRIS - 1) * 3) + 0].pos[2] += (keys[0]) ? -speed : speed;
			//vertData[((NUM_TRIS - 1) * 3) + 1].pos[2] += (keys[0]) ? -speed : speed;
			//vertData[((NUM_TRIS - 1) * 3) + 2].pos[2] += (keys[0]) ? -speed : speed;
			light_z += (keys[0]) ? -speed : speed;
#endif
		}
		if (keys[6] || keys[7]) {
#if SHAPE_MODE == 1
			spheres[2].center[2] += (keys[6]) ? -speed : speed;
			spheresChanged = true;
#elif SHAPE_MODE == 0
			//vertsChanged = true;
			//vertData[((NUM_TRIS - 1) * 3) + 0].pos[1] += (keys[6]) ? -speed : speed;
			//vertData[((NUM_TRIS - 1) * 3) + 1].pos[1] += (keys[6]) ? -speed : speed;
			//vertData[((NUM_TRIS - 1) * 3) + 2].pos[1] += (keys[6]) ? -speed : speed;
			light_y += (keys[6]) ? -speed : speed;
#endif
		}
		if (keys[10] || keys[11]) {
			light_x += (keys[10]) ? -speed : speed;
		}
		if (keys[2] || keys[3]) {
			cam_x += (keys[2]) ? -speed : speed;
		}
		if (keys[4] || keys[5]) {
			cam_y += (keys[4]) ? -speed : speed;
		}
		if (keys[8] || keys[9]) {
			cam_z += (keys[8]) ? -speed : speed;
		}

		if (keys[0] ||
			keys[1] ||
			keys[2] ||
			keys[3] ||
			keys[4] ||
			keys[5] ||
			keys[6] ||
			keys[7] ||
			keys[8] ||
			keys[9] ||
			keys[10] ||
			keys[11] ||
			keys[12]) {

			//forces next frame to render if any changes are made.

			framesRefreshRequired = DEFERRED_REFRESH_SQUARE_DIM * DEFERRED_REFRESH_SQUARE_DIM;
		}

		if (keys[12]) {
			keys[12] = false;
			light_mode = (light_mode == 3) ? 0 : light_mode+1;
		}

		std::chrono::system_clock::time_point cNow = std::chrono::system_clock::now();

		//Every other tick of the given frequency render a frame
		//std::ratio<1, 120> gives 60fps
		//std::ratio<1, 100> gives 50fps
		//std::ratio<1, 60> gives 30fps
		//std::ratio<1, 70> gives 35fps
		//Affects screen refresh and CUDA work
		int ticks = (int)(std::chrono::duration<float, std::ratio<1, 120>>(cNow - WIN_CTIME).count());

		if (((ticks % 2 == 1) && ticks != oldTicks)) {
			WIN_CTIME = cNow;
			oldTicks = ticks;

			uint32_t imageIndex;
			vkAcquireNextImageKHR(device, swapChain,
				std::numeric_limits<uint64_t>::max(),
				imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

			//Vulkan draw first frame
			//CUDA draw all subsequent frames
			if (!startSubmit) {
				submitVulkan(imageIndex);
				startSubmit = 1;
			}
			else {
				submitVulkanCuda(imageIndex);
			}

			VkPresentInfoKHR presentInfo = {};
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

			VkSemaphore signalSemaphores[] = { renderFinishedSemaphore };

			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores = signalSemaphores;

			VkSwapchainKHR swapChains[] = { swapChain };
			presentInfo.swapchainCount = 1;
			presentInfo.pSwapchains = swapChains;
			presentInfo.pImageIndices = &imageIndex;
			presentInfo.pResults = nullptr;  // Optional

			//Draw image to screen
			vkQueuePresentKHR(presentQueue, &presentInfo);

			//TODO: replace this with ray-tracing kernel
			//TODO: Consider Vulkan Compute shader - https://github.com/SaschaWillems/Vulkan/blob/master/examples/raytracing/raytracing.cpp

			//Run CUDA Kernel (waits for render sempaphores to signal)
			cudaUpdateVertexBuffer();

		}

		// Added sleep of 10 millisecs so that CPU does not submit too much work to
		// GPU
		//std::this_thread::sleep_for(std::chrono::microseconds(30000));

		//
		////if it's been a second since last tic
		//if (std::chrono::duration_cast<std::chrono::milliseconds>(cNow - WIN_CTIME).count() >= 1000) {
		//	WIN_CTIME = cNow;
		//}

	}

	void copyBufferToImage(VkCommandBuffer commandBuffer, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkBufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};
		vkCmdCopyBufferToImage(
			commandBuffer,
			buffer,
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&region
		);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		copyBufferToImage(commandBuffer, buffer, image, width, height);

		endSingleTimeCommands(commandBuffer);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion = {};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}
	void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = 0;
		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
		//See: https://vulkan-tutorial.com/Texture_mapping/Images#page_Texture_Image

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		transitionImageLayout(commandBuffer, image, format, oldLayout, newLayout);

		endSingleTimeCommands(commandBuffer);
	}

	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	void createTextureImages() {


		//VkDeviceSize imageSize = WIDTH * HEIGHT * 4 * sizeof(stbi_uc);
		VkDeviceSize imageSize = WIDTH * HEIGHT * sizeof(Texel);

		//createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		//create vulkan buffer (for use with CUDA, hence ext)
#ifdef _WIN64
		if (IsWindows8OrGreater()) {
			createBufferExtMem(imageSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT, stagingBuffer, stagingBufferMemory);
		}
		else {
			createBufferExtMem(imageSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT, stagingBuffer, stagingBufferMemory);
		}
#else
		createBufferExtMem(imageSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, stagingBuffer, stagingBufferMemory);
#endif
		//Image load 1 - texture used by cuda for objects col
		int texWidth, texHeight, texChannels;
		texture1 = (stbi_uc*)stbi_load("Assets/objtexture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

		if (!texture1) {
			throw std::runtime_error("failed to load objtexture.jpg!");
		}

		//Image load 2 - texture used by cuda for objects bump
		texture2 = (stbi_uc*)stbi_load("Assets/objtexturebump.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

		if (!texture2) {
			throw std::runtime_error("failed to load objtexturebump.jpg!");
		}

		//Image load 3 - texture used by cuda for objects bump
		texture3 = (stbi_uc*)stbi_load("Assets/bg.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

		if (!texture3) {
			throw std::runtime_error("failed to load bg.jpg!");
		}

		//Image load ... - texture used by Vulkan (to blit first frame, i.e. instructions)
		if (loadFromFile) {

			pixels = (stbi_uc*)stbi_load("Assets/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

			if (!pixels) {
				throw std::runtime_error("failed to load texture.jpg!");
			}
			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
			memcpy(data, pixels, static_cast<size_t>(imageSize));
			vkUnmapMemory(device, stagingBufferMemory);
		}

		createImage(WIDTH, HEIGHT, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		transitionImageLayout(textureImage, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(WIDTH), static_cast<uint32_t>(HEIGHT));
		transitionImageLayout(textureImage, VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	}

	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void submitVulkan(uint32_t imageIndex) {
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore };
		VkPipelineStageFlags waitStages[] = {
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore,
										  vkUpdateCudaVertexBufSemaphore };

		submitInfo.signalSemaphoreCount = 2;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

	}

	void submitVulkanCuda(uint32_t imageIndex) {
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphore,
										cudaUpdateVkVertexBufSemaphore };
		VkPipelineStageFlags waitStages[] = {
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
		submitInfo.waitSemaphoreCount = 2;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphore,
										  vkUpdateCudaVertexBufSemaphore };

		submitInfo.signalSemaphoreCount = 2;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) !=
			VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}
	}

	void createSyncObjects() {
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
			&imageAvailableSemaphore) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphoreInfo, nullptr,
				&renderFinishedSemaphore) != VK_SUCCESS) {
			throw std::runtime_error(
				"failed to create synchronization objects for a frame!");
		}
	}

	void createSyncObjectsExt() {
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		memset(&semaphoreInfo, 0, sizeof(semaphoreInfo));
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

#ifdef _WIN64
		WindowsSecurityAttributes winSecurityAttributes;

		VkExportSemaphoreWin32HandleInfoKHR
			vulkanExportSemaphoreWin32HandleInfoKHR = {};
		vulkanExportSemaphoreWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
		vulkanExportSemaphoreWin32HandleInfoKHR.pNext = NULL;
		vulkanExportSemaphoreWin32HandleInfoKHR.pAttributes =
			&winSecurityAttributes;
		vulkanExportSemaphoreWin32HandleInfoKHR.dwAccess =
			DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
		vulkanExportSemaphoreWin32HandleInfoKHR.name = (LPCWSTR)NULL;
#endif
		VkExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo = {};
		vulkanExportSemaphoreCreateInfo.sType =
			VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#ifdef _WIN64
		vulkanExportSemaphoreCreateInfo.pNext =
			IsWindows8OrGreater() ? &vulkanExportSemaphoreWin32HandleInfoKHR : NULL;
		vulkanExportSemaphoreCreateInfo.handleTypes =
			IsWindows8OrGreater()
			? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
		vulkanExportSemaphoreCreateInfo.pNext = NULL;
		vulkanExportSemaphoreCreateInfo.handleTypes =
			VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
		semaphoreInfo.pNext = &vulkanExportSemaphoreCreateInfo;

		if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
			&cudaUpdateVkVertexBufSemaphore) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphoreInfo, nullptr,
				&vkUpdateCudaVertexBufSemaphore) != VK_SUCCESS) {
			throw std::runtime_error(
				"failed to create synchronization objects for a CUDA-Vulkan!");
		}
	}

	void cudaVkImportVertexMem() {

		//Description for the import of the VK VertexBuffer as a CUDA object
		cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
		memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));

		//Get handle
#ifdef _WIN64
		cudaExtMemHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalMemoryHandleTypeOpaqueWin32
			: cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
		cudaExtMemHandleDesc.handle.win32.handle = getVkMemHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT
			, stagingBufferMemory);
#else
		cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
		cudaExtMemHandleDesc.handle.fd =
			getVkMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
				, stagingBufferMemory);
#endif

		//cudaExtMemHandleDesc.size = 4 * sizeof(stbi_uc) * WIDTH * HEIGHT;
		cudaExtMemHandleDesc.size = sizeof(Texel) * WIDTH * HEIGHT;

		checkCudaErrors(cudaImportExternalMemory(&cudaExtMemPixelBuffer,
			&cudaExtMemHandleDesc));

		cudaExternalMemoryBufferDesc cudaExtBufferDesc;
		cudaExtBufferDesc.offset = 0;

		//TODO: fix hack assumes texture.jpg is same dimensions as width/height
		//cudaExtBufferDesc.size = 4 * sizeof(stbi_uc) * WIDTH * HEIGHT;
		cudaExtBufferDesc.size = sizeof(Texel) * WIDTH * HEIGHT;

		cudaExtBufferDesc.flags = 0;

		//TODO: replace with "CUDA import VK Image"
		//See: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html
		//Maps a buffer onto the "external memory object"
		checkCudaErrors(cudaExternalMemoryGetMappedBuffer(
			&cudaDevPixelptr, cudaExtMemPixelBuffer, &cudaExtBufferDesc));


		// START IMPORT INTERSECTION RESULTS



		//END IMPORT INTERSECTION RESULTS

		printf("CUDA Imported Vulkan pixel buffer\n");
	}

	//Get access to the Vulkan Semaphore in CUDA
	void cudaVkImportSemaphore() {
		cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
		memset(&externalSemaphoreHandleDesc, 0,
			sizeof(externalSemaphoreHandleDesc));
#ifdef _WIN64
		externalSemaphoreHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32
			: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
		externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
			cudaUpdateVkVertexBufSemaphore);
#else
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
		externalSemaphoreHandleDesc.handle.fd =
			getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
				cudaUpdateVkVertexBufSemaphore);
#endif
		externalSemaphoreHandleDesc.flags = 0;

		checkCudaErrors(cudaImportExternalSemaphore(
			&cudaExtCudaUpdateVkVertexBufSemaphore, &externalSemaphoreHandleDesc));

		memset(&externalSemaphoreHandleDesc, 0,
			sizeof(externalSemaphoreHandleDesc));
#ifdef _WIN64
		externalSemaphoreHandleDesc.type =
			IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32
			: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
		;
		externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(
			IsWindows8OrGreater()
			? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
			: VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
			vkUpdateCudaVertexBufSemaphore);
#else
		externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
		externalSemaphoreHandleDesc.handle.fd =
			getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
				vkUpdateCudaVertexBufSemaphore);
#endif
		externalSemaphoreHandleDesc.flags = 0;
		checkCudaErrors(cudaImportExternalSemaphore(
			&cudaExtVkUpdateCudaVertexBufSemaphore, &externalSemaphoreHandleDesc));
		printf("CUDA Imported Vulkan semaphore\n");
	}

#ifdef _WIN64  // For windows
	HANDLE getVkMemHandle(
		VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType
		, VkDeviceMemory &bufferToGetHandle) {
		HANDLE handle;

		VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
		vkMemoryGetWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
		vkMemoryGetWin32HandleInfoKHR.memory = bufferToGetHandle;
		vkMemoryGetWin32HandleInfoKHR.handleType =
			(VkExternalMemoryHandleTypeFlagBitsKHR)externalMemoryHandleType;

		fpGetMemoryWin32HandleKHR(device, &vkMemoryGetWin32HandleInfoKHR, &handle);
		return handle;
	}
#else
	int getVkMemHandle(
		VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType) {
		if (externalMemoryHandleType ==
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
			int fd;

			VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
			vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
			vkMemoryGetFdInfoKHR.pNext = NULL;
			vkMemoryGetWin32HandleInfoKHR.memory = stagingBufferMemory;
			vkMemoryGetFdInfoKHR.handleType =
				VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

			fpGetMemoryFdKHR(device, &vkMemoryGetFdInfoKHR, &fd);

			return fd;
		}
		return -1;
	}
#endif

#ifdef _WIN64
	HANDLE getVkSemaphoreHandle(
		VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType,
		VkSemaphore& semVkCuda) {
		HANDLE handle;

		VkSemaphoreGetWin32HandleInfoKHR vulkanSemaphoreGetWin32HandleInfoKHR = {};
		vulkanSemaphoreGetWin32HandleInfoKHR.sType =
			VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
		vulkanSemaphoreGetWin32HandleInfoKHR.pNext = NULL;
		vulkanSemaphoreGetWin32HandleInfoKHR.semaphore = semVkCuda;
		vulkanSemaphoreGetWin32HandleInfoKHR.handleType =
			externalSemaphoreHandleType;

		fpGetSemaphoreWin32HandleKHR(device, &vulkanSemaphoreGetWin32HandleInfoKHR,
			&handle);

		return handle;
	}
#else
	int getVkSemaphoreHandle(
		VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType,
		VkSemaphore& semVkCuda) {
		if (externalSemaphoreHandleType ==
			VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
			int fd;

			VkSemaphoreGetFdInfoKHR vulkanSemaphoreGetFdInfoKHR = {};
			vulkanSemaphoreGetFdInfoKHR.sType =
				VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
			vulkanSemaphoreGetFdInfoKHR.pNext = NULL;
			vulkanSemaphoreGetFdInfoKHR.semaphore = semVkCuda;
			vulkanSemaphoreGetFdInfoKHR.handleType =
				VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

			fpGetSemaphoreFdKHR(device, &vulkanSemaphoreGetFdInfoKHR, &fd);

			return fd;
		}
		return -1;
	}
#endif

	void cudaVkSemaphoreSignal(cudaExternalSemaphore_t& extSemaphore) {
		cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
		memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));

		extSemaphoreSignalParams.params.fence.value = 0;
		extSemaphoreSignalParams.flags = 0;
		checkCudaErrors(cudaSignalExternalSemaphoresAsync(
			&extSemaphore, &extSemaphoreSignalParams, 1, streamToRun));
	}

	void cudaVkSemaphoreWait(cudaExternalSemaphore_t& extSemaphore) {
		cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;

		memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));

		extSemaphoreWaitParams.params.fence.value = 0;
		extSemaphoreWaitParams.flags = 0;

		checkCudaErrors(cudaWaitExternalSemaphoresAsync(
			&extSemaphore, &extSemaphoreWaitParams, 1, streamToRun));
	}

	void cudaUpdateVertexBuffer() {

		//Wait until VkUpdateCuda semaphore is signalled
		//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html
		//Signalled by submitVulkan() and submitVulkanCuda()
		cudaVkSemaphoreWait(cudaExtVkUpdateCudaVertexBufSemaphore);

		//CUDA output into Vulkan
		Texel* pixelData = (Texel*)cudaDevPixelptr;

		//checkCudaErrors(cudaMallocManaged((void**)&cudaSpheres, NUM_SPHERES * sizeof(Sphere), cudaMemAttachGlobal));
#if SHAPE_MODE == 1

		if (spheresChanged) {
			spheresChanged = false;
			//Hijack framesRefreshRequired to update frame after sphere data changes
			framesRefreshRequired = DEFERRED_REFRESH_SQUARE_DIM * DEFERRED_REFRESH_SQUARE_DIM;
			checkCudaErrors(
				cudaMemcpyAsync(
					cudaSpheres
					, spheres
					, NUM_SPHERES * sizeof(Sphere)
					, cudaMemcpyHostToDevice
					, streamToRun
				)
			);
		}

/*#elif SHAPE_MODE == 0

		if (vertsChanged) {
			vertsChanged = false;
			//Hijack framesRefreshRequired to update frame after vert data changes
			framesRefreshRequired = DEFERRED_REFRESH_SQUARE_DIM * DEFERRED_REFRESH_SQUARE_DIM;
			checkCudaErrors(
				cudaMemcpyAsync(
					cudaVerts
					, vertData
					, NUM_TRIS * 3 * sizeof(Vertex)
					, cudaMemcpyHostToDevice
					, streamToRun
				)
			);
		}
		*/
#endif

		//Render whole image
		dim3 block(16, 16, 1);
		dim3 grid(WIDTH / (16 * DEFERRED_REFRESH_SQUARE_DIM), HEIGHT / (16 * DEFERRED_REFRESH_SQUARE_DIM), 1);

		//TODO: will not update when spheres changed, i.e. when data asynchronously uploaded with cudaMemcpyAsync()
		if (framesRefreshRequired != 0) {
			
			//make sure to complete the "deferred refresh" cycle in the event of freezeframe, i.e. when no changes to render
			framesRefreshRequired--;

#if PRINT_FPS == 1
			cudaEventRecord(start);
#endif

			get_raytraced_pixels << <grid, block, 0, streamToRun >> > (
				pixelData
				, (BVH*)cudaBVH
				, NUM_OCTREE
				, (Vertex*)cudaVerts
				, NUM_TRIS
				, (Sphere*)cudaSpheres
				, NUM_SPHERES
				, cam_x, cam_y, cam_z
				, light_x, light_y, light_z
				, frameStep
				, DEFERRED_REFRESH_SQUARE_DIM
				, 1.0f
				, light_mode
				,(IntersectionResult*)cudaHitList);

#if PRINT_FPS == 1
			cudaEventRecord(stop);
			//blocks CPU till event is recorded
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\tms frame." << std::endl;
#endif

			//keep count of what sub-frame is being rendered
			if (frameStep != 0) {
				frameStep = ((frameStep) < (DEFERRED_REFRESH_SQUARE_DIM*DEFERRED_REFRESH_SQUARE_DIM)) ? frameStep + 1 : 1;
			}
		}

		//Signal CudaUpdateVk semaphore
		cudaVkSemaphoreSignal(cudaExtCudaUpdateVkVertexBufSemaphore);
	}

	void cleanup() {
		if (enableValidationLayers) {
			DestroyDebugReportCallbackEXT(instance, callback, nullptr);
		}

		vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
		vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
		checkCudaErrors(
			cudaDestroyExternalSemaphore(cudaExtCudaUpdateVkVertexBufSemaphore));
		vkDestroySemaphore(device, cudaUpdateVkVertexBufSemaphore, nullptr);
		checkCudaErrors(
			cudaDestroyExternalSemaphore(cudaExtVkUpdateCudaVertexBufSemaphore));
		vkDestroySemaphore(device, vkUpdateCudaVertexBufSemaphore, nullptr);



		checkCudaErrors(
		cudaEventDestroy(start)
		);
		checkCudaErrors(
		cudaEventDestroy(stop)
		);

		//TODO: Error 30!! Out of bounds memory access? Closes fine without these "frees"
		//wait for async copy, etc. to finish before erasing the device buffer(s)
		/*
		checkCudaErrors(cudaStreamSynchronize(streamToRun));
		#if SHAPE_MODE == 1
			checkCudaErrors(cudaFree(cudaSpheres));
		#elif SHAPE_MODE == 0
			checkCudaErrors(cudaFree(cudaVerts));
		#endif
			checkCudaErrors(cudaFree(cudaTex1));
			checkCudaErrors(cudaFree(cudaTex2));
		*/

		free(vertData);
		free(hitListData);


		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
		
		stbi_image_free(texture1);
		stbi_image_free(texture2);
		stbi_image_free(texture3);
		if (loadFromFile) stbi_image_free(pixels);
		
		vkDestroyImage(device, textureImage, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);

		vkDestroyCommandPool(device, commandPool, nullptr);
		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}
		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}
		
		/*
		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		*/
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		
		vkDestroyBuffer(device, uniformBuffer, nullptr);
		vkFreeMemory(device, uniformBufferMemory, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);
		vkDestroySwapchainKHR(device, swapChain, nullptr);

		checkCudaErrors(cudaDestroyExternalMemory(cudaExtMemPixelBuffer));
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();


	}
};

int main(int argc, char* argv[]) {
	execution_path = argv[0];
	vulkanCudaApp app;

	try {
		app.run();
	}
	catch (const std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}