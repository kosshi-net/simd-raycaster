#define __FILENAME__ "gfx/direct.c"

#include <config.h>

#include "event.h"
#include "direct.h"
#include "gfx.h"
#include "ppm.h"
#include "res.h"
#include "mem.h"


#include <glad/glad.h>

#include <stdio.h>


GLuint direct_vao;

GLuint direct_shader;
GLuint direct_shader_a_coord;
GLuint direct_shader_u_tex;

GLuint direct_vbo;
GLuint direct_tex;



static float* geometry_buffer;

int gfx_direct_init()
{
	logf_info("Init");

	// OPENGL
	glGenVertexArrays(1, &direct_vao);
	glBindVertexArray(direct_vao);

	char* direct_vert_src = res_strcpy( direct_vert_glsl );
	char* direct_frag_src = res_strcpy( direct_frag_glsl );

	direct_shader = gfx_create_shader( direct_vert_src, direct_frag_src );
	mem_free( direct_vert_src );	
	mem_free( direct_frag_src );

	glUseProgram(direct_shader);
	direct_shader_a_coord = glGetAttribLocation (direct_shader, "aCoord");
	direct_shader_u_tex =   glGetUniformLocation(direct_shader, "uTex");


	glGenBuffers( 1, &direct_vbo );
	glEnableVertexAttribArray( direct_shader_a_coord );
	glBindBuffer( GL_ARRAY_BUFFER, direct_vbo );
	glVertexAttribPointer( direct_shader_a_coord, 2, GL_FLOAT, GL_FALSE, 0, NULL );

	// Finish up

	
	glActiveTexture(	GL_TEXTURE0 );
	glGenTextures(		1, &direct_tex );
	glBindTexture(		GL_TEXTURE_2D, direct_tex );

	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	logf_info("Init ok");


	return 0;
}



void gfx_direct_upload( void*buffer, int w, int h ){


	glUseProgram(direct_shader);
	glBindVertexArray(direct_vao);
	
	glActiveTexture(	GL_TEXTURE0 );

	glBindTexture(		GL_TEXTURE_2D, direct_tex );

	glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, 
		GL_RGB, GL_UNSIGNED_BYTE, buffer
	);

	glBindBuffer( 	GL_ARRAY_BUFFER,0 );
	glBindTexture(	GL_TEXTURE_2D, 	0 );

}



int gfx_direct_draw(){

	float min[2] = {-1.0,-1.0};
	float max[2] = { 1.0, 1.0};
	float geom[] = {
		min[0], min[1],
		min[0], max[1],
		max[0], min[1],
		max[0], max[1],
	};
	
	glUseProgram(direct_shader);
	glBindVertexArray(direct_vao);

	glActiveTexture(	GL_TEXTURE0 );
	glBindTexture(		GL_TEXTURE_2D, direct_tex );

	glDisable( GL_DEPTH_TEST );
	glDisable( GL_CULL_FACE );

	glBindBuffer(GL_ARRAY_BUFFER, direct_vbo);

	glBufferData(
		GL_ARRAY_BUFFER, 
		sizeof(geom),
		geom, 
		GL_DYNAMIC_DRAW
	);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	return 0;
}

