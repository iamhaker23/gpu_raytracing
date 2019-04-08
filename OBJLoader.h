#pragma once

#include <stdio.h>
#include <vector>
#include "Vertex.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

struct ObjMat
{
	//Loaded from the File
	char matName[255];
	float ambientCol[3];
	float diffuseCol[3];
	float specCol[3];
	float d;
	float ns;
	int illum;
	char textureName[255] = { 0 };
	char bumpTextureName[255] = { 0 };
	bool bump = false;

	//Created
	int glIndex = 0;
	int glIndexBump = 0;
};

//-------------------------------------------------------------
//- SObjFace
//- Indices for each face of the OBJ file
class SObjFace
{
public:

	unsigned int m_uiVertIdx[3];
	unsigned int m_uiNormalIdx[3];
	unsigned int m_uiTexCoordIdx[3];

	int matId;

	SObjFace(unsigned int vert[3], unsigned int text[3], unsigned int norm[3], int newMatId)
	{
		//copy the data;
		m_uiVertIdx[0] = vert[0] - 1;
		m_uiVertIdx[1] = vert[1] - 1;
		m_uiVertIdx[2] = vert[2] - 1;

		m_uiTexCoordIdx[0] = text[0] - 1;
		m_uiTexCoordIdx[1] = text[1] - 1;
		m_uiTexCoordIdx[2] = text[2] - 1;

		m_uiNormalIdx[0] = norm[0] - 1;
		m_uiNormalIdx[1] = norm[1] - 1;
		m_uiNormalIdx[2] = norm[2] - 1;

		matId = newMatId;
	}

	bool operator<(const SObjFace &face2) const  //Defined to sort list
	{
		if (matId < face2.matId) return true;
		else return false;
	}

};

//Wrapper for vec3 since you cannot store an array directly in an STL vector object
struct vector3d {

	vec3 pos = { 0 };

};

class OBJLoader {

public:
	
	static int loadRawVertexList(const char * fileName, Vertex** vertData, float scale);
	static void loadVertices(Vertex* vertData, int numVerts);
	static int countBVHNeeded(Vertex* vertData, int numVerts, BVH** bvhData);
	static void createBVH(BVH* bvhData, int numBVH, Vertex* vertData, int numVerts);

private:

	static void readTriangleFaceVertTexNorm(char *line, int matId);
	static void readFaceLine(FILE * theFile, int matId);
	static bool myFileLoader(const char *filename);
	static bool myMTLLoader(const char *mainName, const char *filename);
	static int lookupMaterial(char *matName);
	static void splitFrontString(char * inputString, char * frontString, char * restString, int size);

	static std::vector<BVH_BAKE> m_BVH;

	static std::vector<Vertex> m_distinctVerts;
	static std::vector<vector3d> m_vVertices;
	//an STL vector to hold vertex normals
	static std::vector<vector3d> m_vNormals;
	//an STL vector to hold the texture coordinates
	static std::vector<vector3d> m_vTexCoords;
	//an STL vector for the faces
	static std::vector<SObjFace> m_vFaces;
	static std::vector<ObjMat> theMats;

};