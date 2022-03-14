#define __FILENAME__ "game.c"

#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <glad/glad.h>
#include <cglm/cglm.h>

#include <config.h>
#include "cfg.h"

#include "event.h"
#include "ctx.h"
#include "ctx/input.h"

#include "shell.h"

#include "gfx.h"
#include "gfx/crosshair.h"
#include "gfx/text.h"
#include "gfx/direct.h"
#include "gfx/shell.h"
#include "gfx/camera.h"

#include "mem.h"
#include "res.h"
#include "threadpool.h"

#include "cpp/noise.h"

#include <unistd.h>
#include <string.h>
#include <pthread.h>

#include <limits.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double last_time;
static int capped = ' ';
static double last_fps_update = 0;
static int fps_counter = 0;
static int fps = 0;
static struct Camera cam;

#define WIDTH  (800)
#define HEIGHT (600)

void _ss2ws(
	uint16_t x,
	uint16_t max_x,
	uint16_t y,
	uint16_t max_y,
	mat4 mat, // Inverse projection matrix
	vec3 ray
){
	float ray_clip[] = {
		 (x/(float)max_x) * 2.0f - 1.0f,
		-(y/(float)max_y) * 2.0f + 1.0f,
		-1.0f, 1.0f
	};
	
	glm_mat4_mulv3( mat, ray_clip, 1.0f, ray );
	glm_vec3_norm(ray);
}


void command_sensitivity( int argc, char **argv ){
	if(argc == 2) cfg_get()->sensitivity = atof(argv[1]);
	logf_info("%f", cfg_get()->sensitivity);
}
void command_speed( int argc, char **argv ){
	if(argc == 2) cfg_get()->speed = atof(argv[1]);
	logf_info("%f", cfg_get()->speed);
}

void command_fov( int argc, char **argv ){
	if(argc == 2) cfg_get()->fov = atof(argv[1]);
	logf_info("%f", cfg_get()->fov);
}



typedef uint8_t Voxel;

static Voxel *tree[32];
static uint32_t root = 128;
static uint8_t  bitw;
static uint32_t tree_count[32];
static uint32_t tree_start[32];
static uint32_t tree_total;
static uint32_t tree_bitw[32];
static Voxel *tree_flat;

void print_coords(uint32_t i, uint32_t bitw)
{
	logf_info( "x %i", i >> bitw*0 & ((1<<bitw*0)-1) );
	logf_info( "y %i", i >> bitw*1 & ((1<<bitw*1)-1) );
	logf_info( "z %i", i >> bitw*2 & ((1<<bitw*2)-1) );
}

void tree_init(){

	bitw = __builtin_ctz(root);


	uint32_t precomp_size = (0x249249249) & ( (1<<(bitw+1)*3) - 1 );
	tree_total = precomp_size;
	logf_info("Alloc %i", precomp_size);
	tree_flat = mem_calloc( precomp_size );
	

	tree_count[0] = root*root*root;
	tree_bitw[0] = bitw;

	uint32_t total = 0;
	for( int h = 0; h < bitw+1; h++ ){
		tree[h] = tree_flat + total;
		tree_start[h] = total;

		logf_info("Init, pointer %i,  Height %i, count %i, bitw %i", total, h, tree_count[h], tree_bitw[h]);
		tree_count[h+1] = tree_count[h]>>3;
		tree_bitw[h+1]  = tree_bitw[h]-1;
	
		total += tree_count[h];

		if( total > precomp_size ) logf_error("OUT OF BOUNDS!!?");
	}
		
	if(0)
	for( uint32_t i = 0; i < tree_count[0]; i++ ){

		uint32_t mask = (1<<tree_bitw[0])-1;
		int x = i >> tree_bitw[0]*0 & mask;
		int y = i >> tree_bitw[0]*1 & mask;
		int z = i >> tree_bitw[0]*2 & mask;
		
		float freq = 2.0;
		if(noise_u_simplex_f3(
			x*freq,
			y*freq,
			z*freq
		) < 0.85 ) continue;

		tree[0][i] = rand();
	}else{
		float   *data = (float*)res_file(bunny);
		uint32_t items = res_size(bunny) / sizeof(float);

		for( int i = 0; i < items; i += 6 ){
			float middle = root/2.0;
			
			//printf("%f %f %f\n", data[i+0], data[i+1], data[i+2]);
			uint32_t x = middle + (data[i+0] * (1<<9)); 
			uint32_t y =          (data[i+1] * (1<<9)); 
			uint32_t z = middle + (data[i+2] * (1<<9));
			
			if(root < (x|y|z) ) {
				printf("Out of range %i<%i %i %i %i\n", root, x|y|z, x, y, z);
				continue;
			};

			int b = tree_bitw[0];
			uint32_t k = (z << b | y) << b | x;

			tree[0][k] = (64) + (32)*(rand() / (float)RAND_MAX);
		}
	}


	// h is height in octree, 0 being leaf
	for( uint32_t h = 0; h < bitw; h++ ) {
		logf_info("Height %i, count %i, bitw %i", h, tree_count[h], tree_bitw[h]);
		uint32_t mask = (1<<tree_bitw[h])-1;
		#pragma omp parallel for
		for( uint32_t i = 0; i < tree_count[h]; i++ ){
			Voxel vxl = tree[h][i];
			uint32_t k = 0;
			int x = i >> tree_bitw[h]*0 & mask;
			int y = i >> tree_bitw[h]*1 & mask;
			int z = i >> tree_bitw[h]*2 & mask;
			x>>=1;
			y>>=1;
			z>>=1;
			int b = tree_bitw[h+1];
			k = (z << b | y) << b | x;
			if( k >= tree_count[h+1] ) logf_error("Out of bounds");
			if(vxl) tree[h+1][k] = vxl;
		}
	}

	
}


void cast_floats( float*origin, float*v, uint8_t*c ){
	float 	 p[3];
	uint32_t sign[3];
	float    div[3];

	for( int i = 0; i < 3; i++ ){
		p[i] = origin[i];
		sign[i] = v[i] > 0;
		div[i] = (1.0/v[i]);
	}

	int limit = 256;
	int height = bitw-3;
	while(limit--){

		uint32_t tp[3];
		for( int i = 0; i < 3; i++ )
			tp[i] = (int)p[i] >> height;

		uint32_t b = tree_bitw[height];
		uint32_t i = (tp[2] << b | tp[1]) << b | tp[0];
		Voxel vxl = tree[height][i];

		if( vxl != 0) {
			if( height == 0 ) {
				for( int i = 0; i < 3; i++ ) c[i] = limit - 16 + ( vxl/16 );
				return;
			}
			height--;
			continue;
		}
		
		uint32_t unit = 1 << height;
		int32_t rect[3];
		float t = 99.0;

		for( int i = 0; i < 3; i++ ){
			rect[i] = tp[i] << height;
			rect[i] += unit * sign[i];
			float nt = (rect[i] - p[i]) * div[i];
			t = (nt < t) ? nt : t;
		}
		t += 0.1;
		uint32_t exit = 0;
		for( int i = 0; i < 3; i++ ){
			p[i] += v[i] * t;
			exit += ( p[i] < 0 || p[i]+1 > root);
		}
		if(exit){	
			c[0] = 255-limit; c[1] = 0x0; c[2] = 255-limit;
			return;
		}
		height++;
	}

	c[0] = 0xFF; c[1] = 0x0; c[2] = 0x0;
	return;
}




void cast( float*fp, float*fv, uint32_t*c ){
	int32_t fixed_bitw  = 8;
	int32_t fixed_point = 1<<fixed_bitw;

	const int DIVMANT = 16;

	
	int32_t p    [3];		
	int32_t v    [3];		
	int32_t tv   [3];		
	int32_t vdiv [3];		
	int32_t usign[3];	
	int32_t tp   [3]; 

	for( int i = 0; i < 3; i++ ){
		p[i] = (int32_t)(fixed_point * fp[i]) & ((root<<fixed_bitw)-1);
		v[i] = fixed_point * fv[i];

		vdiv[i] = (1.0 / (float)(v[i]-0.5) * (float)(1<<DIVMANT))+0.5;

		if(v[i] == 0) v[i]--;
		usign[i] = (v[i]>0);
	}
	int32_t height = bitw-2;


	for( int i = 0; i < 16; i++ ) c[i] = 0;

	for ( int _i = 0; _i < 512; _i++ ){

		int32_t grid_shift = fixed_bitw+height;
		uint32_t b = bitw-height;

		for( int i = 0; i < 3; i++ ){
			tp[i] = p[i] >> grid_shift;
		}

		uint32_t i = (tp[2] << b | tp[1]) << b | tp[0];
		
		Voxel vxl = tree[height][i];

		if( vxl != 0) {
			if(height == 0) {
				//*c = _i << 16 | vxl;
				*c = _i;
				return;
			}
			height--;
			continue;
		}
		

		int32_t t = INT_MAX;
		for( int i = 0; i < 3; i++ ){
			tp[i] = (tp[i] + usign[i] ) << grid_shift;
			tv[i] = ( (tp[i] - p[i]) ) * ( vdiv[i] );
		}
		t = MIN( tv[0], MIN( tv[1], tv[2] ) );


		t+=(1<<11);
		
		for( int i = 0; i < 3; i++ ){
			p[i] += v[i] * t >> DIVMANT;
			p[i] &= (root << fixed_bitw) - 1 ;
		}
		height++;
	}

	//c[0] = 0x0; c[1] = 0x0; c[2] = 0x0;
	return;
}


#include <emmintrin.h>
#include <immintrin.h>

#define aint32_t  __attribute__ ((aligned(64))) int32_t
#define auint32_t __attribute__ ((aligned(64))) uint32_t

#define TILEX 4
#define TILEY 2
#define BATCH (TILEX*TILEY)
void cast_avx2( float*fp, float*fv, uint32_t*c ){

	int32_t mantissa_bits = 8;
	int32_t mantissa = 1 << mantissa_bits;
	
	const int DIVMANT =  16; // Accuracy of rcp

	aint32_t p  [ BATCH*3 ]; // Position 
	aint32_t v  [ BATCH*3 ]; // Velocity 
	aint32_t sm [ BATCH*3 ]; // Sign mask
	aint32_t rcp[ BATCH*3 ]; // Reciprocal of velocity

	for( int i = 0; i < BATCH*3; i++ ){
		p[i] = (int32_t)(mantissa * fp[i]) & ((root<<mantissa_bits)-1);
		v[i] = mantissa * fv[i];
		rcp[i] = (1.0 / (v[i]-0.5) * (float)(1 << DIVMANT)) + 0.5;
		if(v[i] == 0) v[i]--;
		sm[i] = (v[i]>0);
	}
	
	__m256i p_x = _mm256_load_si256( (__m256i*)( p + 0 ) );
	__m256i p_y = _mm256_load_si256( (__m256i*)( p + BATCH ) );
	__m256i p_z = _mm256_load_si256( (__m256i*)( p + BATCH*2 ) );

	__m256i v_x = _mm256_load_si256( (__m256i*)( v + 0 ) );
	__m256i v_y = _mm256_load_si256( (__m256i*)( v + BATCH ) );
	__m256i v_z = _mm256_load_si256( (__m256i*)( v + BATCH*2 ) );

	__m256i sm_x = _mm256_load_si256( (__m256i*)( sm + 0 ) );
	__m256i sm_y = _mm256_load_si256( (__m256i*)( sm + BATCH ) );
	__m256i sm_z = _mm256_load_si256( (__m256i*)( sm + BATCH*2 ) );

	__m256i rcp_x = _mm256_load_si256( (__m256i*)( rcp + 0 ) );
	__m256i rcp_y = _mm256_load_si256( (__m256i*)( rcp + BATCH ) );
	__m256i rcp_z = _mm256_load_si256( (__m256i*)( rcp + BATCH*2 ) );

	__m256i h   = _mm256_set1_epi32( bitw-2 );

	__m256i _scalar;

	__m256i grid_shift; // Shift for fixed point to grid and back
	__m256i grid_bitw;  // log2(root)

	// Temporary point
	__m256i tp_x; 
	__m256i tp_y;
	__m256i tp_z;
	
	// Time vector 
	__m256i tv_x; 
	__m256i tv_y;
	__m256i tv_z;
	__m256i t;

	__m256i index; 
	__m256i mask;
	__m256i hmask;
	__m256i inv_mask;

	__m256i voxel;

	__m256i color = _mm256_set1_epi32( 0 );

	__m256i done = _mm256_set1_epi32( UINT_MAX );

	for( int _i = 0; _i < 512; _i++ ){

		// grid_shift = mantissa_bits + h
		_scalar    = _mm256_set1_epi32( mantissa_bits );
		grid_shift = _mm256_add_epi32 ( _scalar, h );

		// grid_bitw  = bitw - h
		_scalar    = _mm256_set1_epi32( bitw );
		grid_bitw  = _mm256_sub_epi32 ( _scalar, h );
	
		// tp = p >> grid_shift;
		tp_z       = _mm256_srlv_epi32 ( p_z, grid_shift );
		tp_y       = _mm256_srlv_epi32 ( p_y, grid_shift );
		tp_x       = _mm256_srlv_epi32 ( p_x, grid_shift );

		// index   = (tp_z << grid_bitw | tp_y) << grid_bitw | tp_x;
		index      = _mm256_sllv_epi32 ( tp_z,  grid_bitw );
		index      = _mm256_or_si256   ( index, tp_y );
		index      = _mm256_sllv_epi32 ( index, grid_bitw );
		index      = _mm256_or_si256   ( index, tp_x );

		_scalar = _mm256_set1_epi32( 0 );

		__m256i index_offsets = _mm256_mask_i32gather_epi32( 
			_scalar,
			tree_start, 
			h, 
			done, 
			sizeof(uint32_t) 
		);

		index = _mm256_add_epi32( index, index_offsets );

		voxel = _mm256_mask_i32gather_epi32( 
			_scalar, 
			tree_flat,
			index, 
			done, 
			sizeof(Voxel)
		);

		_scalar = _mm256_set1_epi32( 0xFF );
		voxel = _mm256_and_si256( voxel, _scalar );

		_scalar = _mm256_set1_epi32( 0 );
		mask     = _mm256_cmpeq_epi32( voxel, _scalar );
		inv_mask = _mm256_cmpgt_epi32( voxel, _scalar );
	
		hmask = _mm256_cmpgt_epi32( h, _scalar );
		hmask = _mm256_or_si256( hmask, mask );

		// write color if not done
		if(0){
			_scalar = _mm256_set1_epi32( 1 );
			_scalar = _mm256_and_si256(  _scalar, done );
			color = _mm256_add_epi32( color, _scalar );
		}else{
			
			_scalar = _mm256_set1_epi32( (_i>>2)<<16 );
			_scalar = _mm256_add_epi32 ( _scalar, voxel );
			_scalar = _mm256_or_si256 ( _scalar, voxel );

			_scalar = _mm256_and_si256( _scalar, inv_mask );
			color = _mm256_and_si256( color, mask );
			color = _mm256_add_epi32( color, _scalar);
		}
		

		done =  _mm256_and_si256( hmask, done );

		_scalar = _mm256_set1_epi32( ( 1 ) );
		_scalar = _mm256_and_si256( _scalar, inv_mask );
		h       = _mm256_sub_epi32( h, _scalar );

		if( _mm256_testz_si256(done, _mm256_set1_epi32(0xFFFF)) ) break;

		// tp = (tp + sm ) << grid_shift
		tp_x = _mm256_add_epi32( tp_x, sm_x );
		tp_y = _mm256_add_epi32( tp_y, sm_y );
		tp_z = _mm256_add_epi32( tp_z, sm_z );

		tp_x = _mm256_sllv_epi32(tp_x, grid_shift );
		tp_y = _mm256_sllv_epi32(tp_y, grid_shift );
		tp_z = _mm256_sllv_epi32(tp_z, grid_shift );

		// tv = (tp - p) * rcp;
		tv_x = _mm256_sub_epi32(   tp_x, p_x );
		tv_y = _mm256_sub_epi32(   tp_y, p_y );
		tv_z = _mm256_sub_epi32(   tp_z, p_z );

		tv_x = _mm256_mullo_epi32( tv_x, rcp_x );
		tv_y = _mm256_mullo_epi32( tv_y, rcp_y );
		tv_z = _mm256_mullo_epi32( tv_z, rcp_z );

		// t = min(tv.x, tv.y, tv.z)
		t    = _mm256_min_epi32( tv_x, tv_y );
		t    = _mm256_min_epi32( t,    tv_z );
		
		// t += epsilon
		// Tradeoff between voxel size and diagonal line artifacts
		_scalar = _mm256_set1_epi32( ( (1<<9) ) ); // Original value: 11
		t       = _mm256_add_epi32( t, _scalar );

		// p += (v * t) >> DIVMANT; 
		t = _mm256_and_si256( t, mask );
		
		tp_x = _mm256_mullo_epi32 (  v_x, t );
		tp_y = _mm256_mullo_epi32 (  v_y, t );
		tp_z = _mm256_mullo_epi32 (  v_z, t );

		tp_x = _mm256_srai_epi32( tp_x, DIVMANT );
		tp_y = _mm256_srai_epi32( tp_y, DIVMANT );
		tp_z = _mm256_srai_epi32( tp_z, DIVMANT );

		p_x  = _mm256_add_epi32(  p_x, tp_x ); 
		p_y  = _mm256_add_epi32(  p_y, tp_y ); 
		p_z  = _mm256_add_epi32(  p_z, tp_z ); 
		
		// p = p & (root<<mantissa_bits)-1
		_scalar = _mm256_set1_epi32( ( root << mantissa_bits ) - 1 );
		p_x  = _mm256_and_si256( p_x, _scalar ); 
		p_y  = _mm256_and_si256( p_y, _scalar ); 
		p_z  = _mm256_and_si256( p_z, _scalar ); 

		// h++;
		_scalar = _mm256_set1_epi32( ( 1 ) );
		_scalar = _mm256_and_si256( _scalar, mask );
		h       = _mm256_add_epi32( h, _scalar );
	}

	_mm256_store_si256( (__m256i*)(c), color );
}

static const char *renderer;
static uint8_t *image;

int game_init(){

	shell_bind_command("sensitivity", 			&command_sensitivity);
	shell_bind_command("speed", 				&command_speed);
	//shell_bind_command("fov", 					&command_fov);

	cam.yaw = 90.0+45.0;
	cam.location[0] = 32.0;
	cam.location[1] = 32.0;
	cam.location[2] = 32.0;
	cam.fov = cfg_get()->fov;
	cam.far_plane = 1024*2;
	cfg_get()->speed = 10.0;
	renderer = (char*)glGetString(GL_RENDERER);
	last_time = ctx_time();
	gfx_direct_init();
	image = mem_alloc( WIDTH*HEIGHT * 3 );
	tree_init();

	return 0;
}

static int mode = 0;

void game_tick(){

	double now = ctx_time();
	double delta = now - last_time;

	fps_counter++;
	if( now > last_fps_update+1.0 ){
		last_fps_update = now;
		fps = fps_counter;
		fps_counter = 0;
	}

	int width, height;
	ctx_get_window_size( &width, &height  );
	glViewport(0,0,width,height);
	
	//
	// Camera position and matrices
	//

	cam.far_plane = cfg_get()->draw_distance;
	cam.fov =       cfg_get()->fov;

	double cur_x, cur_y;
	ctx_cursor_movement( &cur_x, &cur_y  );
	cam.yaw 	-= cur_x * cfg_get()->sensitivity;
	cam.pitch 	-= cur_y * cfg_get()->sensitivity;
	
	if( cam.pitch > 90   ) cam.pitch=90;
	if( cam.pitch < -90  ) cam.pitch=-90;

	float pitch = 	glm_rad(-cam.pitch);
	float yaw = 	glm_rad(-cam.yaw);

	float s = delta*cfg_get()->speed;

	float dir[] = {
		cosf(pitch) * sinf(yaw),
		sinf(-pitch),
		cosf(pitch) * cosf(yaw)
	};

	float right[] = {
		sinf(yaw + 3.14/2), 
		0,
		cosf(yaw + 3.14/2)
	};
		
	int vx = 0, vy = 0;	
	ctx_movement_vec( &vx, &vy );
	
	for( int i = 0; i < 3; i++  )
		cam.location[i] -= ( dir[i]*vx + right[i]*vy )*s;

	glm_perspective( glm_rad(cam.fov), width/(float)height, 0.1, cam.far_plane, cam.projection );
	glm_mat4_inv( cam.projection, cam.inverse_projection  );
	glm_mat4_identity(cam.view);
	glm_mat4_identity(cam.view_rotation);
	glm_mat4_identity(cam.view_translation);

	float location_inverse[3];
	for (int i = 0; i < 3; ++i)
		location_inverse[i] = -cam.location[i];

	glm_translate(cam.view_translation, location_inverse);
	glm_rotate( cam.view_rotation, glm_rad(cam.pitch), (vec3){1, 0, 0} );
	glm_rotate( cam.view_rotation, glm_rad(cam.yaw)  , (vec3){0, 1, 0} );
	glm_mul( 
		cam.view_rotation,
		cam.view_translation,
		cam.view
	 );
	glm_mul( cam.projection, cam.view, cam.view_projection );
	glm_mul( 
		cam.projection,
		cam.view_rotation,
		cam.inv_proj_view_rot
	);
	glm_mat4_inv(
		cam.inv_proj_view_rot,
		cam.inv_proj_view_rot
	);

	glClearColor( 0.5, 0.5, 0.7, 1.0  );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	if(ctx_input_place_block()) mode = !mode;
	
	for( int x = 0; x < WIDTH/TILEX; x++ ){
#pragma omp parallel for
		for( int y = 0; y < HEIGHT/TILEY; y++ ){
		
			float   rays   [ BATCH * 3 ];
			float   origins[ BATCH * 3 ];
			auint32_t color[ BATCH     ] = {0};

			for( int t = 0; t < BATCH; t++ ){

				int rx = x*TILEX + (t&3);
				int ry = y*TILEY + (t>>2);
				
				vec3 ray;
				_ss2ws( rx, WIDTH, ry, HEIGHT,
					cam.inv_proj_view_rot,
					ray
				);

				for( int i = 0; i < 3; i++ ){
					rays    [ t + i*BATCH ] = ray[i];
					origins [ t + i*BATCH ] = cam.location[i];
				}
				if(mode) cast( cam.location, ray, color + t );
			}
			if(!mode)
				cast_avx2( origins, rays, color );

			for( int t = 0; t < BATCH; t++ ){
				int rx = x*TILEX + (t&3);
				int ry = y*TILEY + (t>>2);
				int i = (HEIGHT-ry-1)*WIDTH + rx;
				
				uint8_t r = color[t]       & 0xFF;
				uint8_t g = color[t] >> 8  & 0xFF;
				uint8_t b = color[t] >> 16 & 0xFF;

				float a = b/256.0;
				float a_inv = 1.0-a;
				r = r * a_inv + 255 * a;

				if(b == 0xFF) {
					r=0;
					b=0;
				}
				image[ i*3+0 ] = r;
				image[ i*3+1 ] = g;
				image[ i*3+2 ] = b * 0.35;
			}
		}
	}
	
	gfx_direct_upload( image, WIDTH, HEIGHT );
	gfx_direct_draw();
	

	//
	// Info
	//

	int w, h;
	ctx_get_window_size( &w, &h );
	float sx = 2.0 / w;
	float sy = 2.0 / h;

	float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};

	glClear( GL_DEPTH_BUFFER_BIT  );

	static char charbuf[512];

	snprintf( charbuf, 512, 	
			"%i%c FPS  %.1fms\n"
			"Memory    %li / %li MB\n"
			"%ix%i %s\n"
			"X: %.2f\n"
			"Y: %.2f\n"
			"Z: %.2f\n",
		
		fps, capped, delta*1000,
		
		mem_dynamic_allocation() >> 20,
		cfg_get()->heap >> 20,
		WIDTH, HEIGHT,
		(mode) ? "scalar" : "AVX2",
		cam.location[0], cam.location[1], cam.location[2]
	);

	gfx_text_draw( 
		charbuf,
		-1 + 8 * sx,
		 1 - 20 * sy,
		sx, sy,
		color, FONT_SIZE
	);

	gfx_crosshair_draw();

	glFlush();

	// Frame limit
	capped = ' ';
	double worked = ctx_time();
	if( worked-now < 1.0/500.0 ) {
		while( ctx_time() < now+0.0032 ) usleep(50);
		capped = '*';
	}

	last_time = now;
}


