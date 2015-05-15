/*
 * mask_bmp_2D.h
 *
 *  Copyright (C) 2015 Diamond Light Source
 *
 *  Author: Luis Fuentes-Montero (Luiso)
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */

#ifndef DIALS_MASK_LOW_LEVEL_H
#define DIALS_MASK_LOW_LEVEL_H
#define PX_SCALE 85
#define DST_BTW_LIN 7
#include <iostream>
#include <string>
#include <cmath>
#include <scitbx/array_family/flex_types.h>

using scitbx::af::flex_double;
using scitbx::af::flex_int;
using scitbx::af::flex_grid;

int get_mask_img_array( int (&mask_bw_img)[PX_SCALE][PX_SCALE][4]){
  int err_cod = 0;

  // cleaning mask
  for(int dpt = 0; dpt < DST_BTW_LIN; dpt++){
    for(int row = 0; row < PX_SCALE; row++){
      for(int col = 0; col < PX_SCALE; col++){
        mask_bw_img[col][row][dpt] = 0;
      }
    }
  }

  // painting diagonal lines from left top to right bottom
  for(int row = 0; row < PX_SCALE; row++){
    for(int col = 0; col < PX_SCALE; col++){
      for(int dg_pos = -84; dg_pos < 85; dg_pos += DST_BTW_LIN){
        if(row == col + dg_pos){
          mask_bw_img[col][row][0] = 1;
        }
      }
    }
  }

  // painting diagonal lines from left bottom to right top
  for(int row = 0; row < PX_SCALE; row++){
    for(int col = 0; col < PX_SCALE; col++){
      for(int dg_pos = 0; dg_pos < 175; dg_pos += DST_BTW_LIN){
        if(row == -col + dg_pos){
          mask_bw_img[col][row][1] = 1;
        }
      }
    }
  }

  // painting horizontal lines
  for(int row = 0; row < PX_SCALE; row++){
    for(int col = 0; col < PX_SCALE; col += DST_BTW_LIN){
        mask_bw_img[col][row][2] = 1;
    }
  }

  // painting vertical lines
  for(int row = 0; row < PX_SCALE; row += DST_BTW_LIN ){
    for(int col = 0; col < PX_SCALE; col++ ){
        mask_bw_img[col][row][3] = 1;
    }
  }

  std::cout << "\n Hi from mask_bmp_2D\n";

  return err_cod;
}


#endif
