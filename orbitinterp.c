/**************************************************************************
* FUNCTION: ADDGPSORB2OGDR
* Purpose:
* To add the GPS-based orbit to an OGDR
*
**************************************************************************/
#include "orbitinterp.h"

int main(int argc, char *argv[])
{
   double ellipsoid_axis, ellipsoid_flattening, taimutc, gpsmutc;
   double *otime;
   double lat_scale_factor, lon_scale_factor, alt_scale_factor, alt_add_offset;
   double tgps, xyz[3], llh[3], xyzdot[3];
   double prodalt, prodlat, prodlon, orbalt, orblat, orblon;
   int32_t i, incfile, iposfile, norb;
   int ncid, data01id, time_dimid;
   int time_id, lat_id, lon_id, alt_id;
   int    alt_fillvalue;
   int    *alt;
   int    *lat;
   int    *lon;
   static double torb[MAXORB], Xorb[MAXORB], Yorb[MAXORB], Zorb[MAXORB];
   size_t time_len;
   static size_t start[1] = {0};

/* Check for insufficient arguments
*/
   if (argc != 5) {
     fprintf(stdout,"%s -ncfile netCDFFile -posgoafile posgoafile\n", argv[0]);
     exit(1);
   }

   incfile  = 0;
   iposfile = 0;
   for (i = 1; i < argc; ++i) {
     if (strcmp(argv[i],"-ncfile") == 0) {
       ++i;
       if (i >= argc) {
         break;
       }
       incfile = i;
     }
     if (strcmp(argv[i],"-posgoafile") == 0) {
       ++i;
       if (i >= argc) {
         break;
       }
       iposfile = i;
     }
   }
   if (incfile == 0) {
     fprintf(stdout,"-ncfile option required to specify input netCDF file\n");
     exit(1);
   }
   if (iposfile == 0) {
     fprintf(stdout,"-posgoafile option required to specify input orbit file in pos_goa format\n");
     exit(1);
   }

/* Read the orbit file in pos_goa format
*/
   if (readposgoafile(torb, Xorb, Yorb, Zorb, &norb, argv[iposfile], MAXORB) == 1) {
     fprintf(stderr,"ERROR: orbitinterp has error reading pos_goa file %s\n", argv[iposfile]);
     exit(1);
   }
   fprintf(stdout,"%d orbit records found in %s\n", norb, argv[iposfile]);

/* Open the netCDF file
*/
   NCERR(nc_open(argv[incfile], NC_NOWRITE, &ncid), argv[incfile] );
   data01id = ncid;

/* Read the reference ellipsoid parameters used to compute lat/lon/alt from this netCDF file
*/
   NCERR( nc_get_att_double(ncid, NC_GLOBAL, "ellipsoid_axis", &ellipsoid_axis), argv[incfile]);
   NCERR( nc_get_att_double(ncid, NC_GLOBAL, "ellipsoid_flattening", &ellipsoid_flattening), argv[incfile]);
   fprintf(stdout,"Reference ellipsoid parameters from netCDF file.\n");
   fprintf(stdout,"  ellipsoid_axis       = %f\n", ellipsoid_axis);
   fprintf(stdout,"  ellipsoid_flattening = %16.12f\n", ellipsoid_flattening);

/* For a GDR-F format file uncomment this line
   NCERR( nc_inq_grp_ncid(ncid, "data_01", &data01id), args[incfile]);
*/

/* Read the dimension of netCDF time contents
*/
   NCERR( nc_inq_dimid(data01id, "time", &time_dimid), argv[incfile]);
   NCERR( nc_inq_dimlen(data01id, time_dimid, &time_len), argv[incfile]);
   fprintf(stdout,"%d time records in netCDF file\n", time_len, argv[incfile]);

/* Read the time and lat/lon/alt from the netCDF file (used only for reference)
*/
   otime   = malloc(time_len * sizeof(double));
   lat     = malloc(time_len * sizeof(int));
   lon     = malloc(time_len * sizeof(int));
   alt     = malloc(time_len * sizeof(int));

   NCERR( nc_inq_varid(data01id, "time", &time_id), argv[incfile]);
   NCERR( nc_get_vara_double(data01id, time_id, start, &time_len, otime), argv[incfile]);

/* Read leap second information
*/
   NCERR( nc_get_att_double(data01id, time_id, "tai_utc_difference", &taimutc), argv[incfile]);
   gpsmutc = GPSMTAI - taimutc;
   fprintf(stdout,"gpsmutc = %f\n", gpsmutc);

   NCERR( nc_inq_varid(data01id, "lat", &lat_id), argv[incfile]);
   NCERR( nc_get_vara_int(data01id, lat_id, start, &time_len, lat), argv[incfile]);
   NCERR( nc_get_att_double(data01id, lat_id, "scale_factor", &lat_scale_factor), argv[incfile]);

   NCERR( nc_inq_varid(data01id, "lon", &lon_id), argv[incfile]);
   NCERR( nc_get_vara_int(data01id, lon_id, start, &time_len, lon), argv[incfile]);
   NCERR( nc_get_att_double(data01id, lon_id, "scale_factor", &lon_scale_factor), argv[incfile]);

   NCERR( nc_inq_varid(data01id, "alt", &alt_id), argv[incfile]);
   NCERR( nc_get_vara_int(data01id, alt_id, start, &time_len, alt), argv[incfile]);
   NCERR( nc_get_att_double(data01id, alt_id, "scale_factor", &alt_scale_factor), argv[incfile]);
   NCERR( nc_get_att_double(data01id, alt_id, "add_offset", &alt_add_offset), argv[incfile]);
   NCERR( nc_get_att_int(data01id, alt_id, "_FillValue", &alt_fillvalue), argv[incfile]);

/* Loop through every data record in netCDF file
*/
   for (i = 0; i < time_len; ++i) {

/* Convert data time from netCDF file to "GPS time" used in orbit file
*  netCDF file counts UTC seconds since Jan 1, 2000 00:00:00 UTC
*  posgoa file uses GPS seconds since Jan 1, 2000 12:00:00 GPS
*/
      tgps = otime[i] - 43200.0 + gpsmutc;

/* Convert altitude on product to absolute value
*  Convert product lat/lon to degrees
*/
      prodalt = ((double) alt[i])*alt_scale_factor + alt_add_offset;
      prodlat = ((double) lat[i])*lat_scale_factor;
      prodlon = ((double) lon[i])*lon_scale_factor;

/* Interpolate the orbit and compute lat/lon/alt
*/
      if (interporb(xyz, llh, xyzdot, tgps, ellipsoid_axis, ellipsoid_flattening,
                    torb, Xorb, Yorb, Zorb, norb) == 0) {
        orbalt    = llh[2];
        orblat    = llh[0];
        orblon    = llh[1];
      }
      else {
        fprintf(stdout,"Error interpolating orbit at product time = %d\n", otime[i]);
        continue;
      }

/* Print netCDF data record number, lat, lon, alt, orbit file lat, lon, alt, differences of lat, lon, alt
*/
      fprintf(stdout,"%6d  %12.6f  %12.6f  %15.6f  %12.6f  %12.6f  %15.6f  %12.6f  %12.6f  %12.6f\n", i, prodlat, prodlon, prodalt, orblat, orblon, orbalt, prodlat-orblat, prodlon-prodlon, prodalt-orbalt);
   }

/* Close the netCDF file
*/
   NCERR ( nc_close(ncid), argv[incfile] );

/* Free memory
*/
   free(otime);
   free(lat);
   free(lon);
   free(alt);

   return 0;
}
/**************************************************************************
* FUNCTION: READPOGOAFILE
* Purpose:
* To read a pos-goa file
* 
* Input:
*   posgoafile[] - Name of JPL pos_goa file with orbit
*   ndim         - Array size of torb, Xorb, Yorb, Zorb
* Output:
*   torb[]       - Array of time tags (UTC time) J2000 sec
*   Xorb[]       - Array of X coordinates (m)
*   Yorb[]       - Array of X coordinates (m)
*   Zorb[]       - Array of X coordinates (m)
*   norb         - Number of points in orbit array
**************************************************************************/
int readposgoafile(double torb[], double Xorb[], double Yorb[], double Zorb[],
                   int32_t *norb, char posgoafile[], int32_t ndim)
{
    double  t, x, y, z;
    int32_t i, it;
    char    line[500], frame[10], satname[20];
    FILE    *ifin;

    if ((ifin = fopen(posgoafile,"r")) == NULL) {
      fprintf(stderr,"ERROR: Could not open file %s\n", posgoafile);
      return 1;
    }
    i = 0;
    while (fgets(line,MAXLINE,ifin) != NULL) {
      sscanf(line,"%s %s %d %lf %lf %lf %lf", frame, satname, &it, &t, &x, &y, &z);
      if (i >= ndim) {
        fprintf(stderr,"ERROR: Reset orbit dimension to > %d\n", ndim);
	fclose(ifin);
	return 1;
      }
      torb[i] = ((double) it) + t;
      Xorb[i] = 1.0e3*x;
      Yorb[i] = 1.0e3*y;
      Zorb[i] = 1.0e3*z;
      i       = i + 1;
    }
    fclose(ifin);
    *norb = i;
}
/**************************************************************************
* FUNCTION: NCERR
* Purpose: To check for and handle errors with interfacing with NetCDF
* files
*
* Input:
*   status - Status from NetCDF command
**************************************************************************/
int NCERR(int status, char ogdrfile[]) {

  if (status != NC_NOERR) {
    fprintf(stderr,"ERROR: %s\n", nc_strerror(status));
    fprintf(stderr,"ERROR: addgpsorb2ogdr could not process OGDR %s\n", ogdrfile);
    exit(2);
  }

}
/**************************************************************************
* FUNCTION: INTERPORB
* Purpose:
* To interpolate an orbit given an array of x,y,z, coordinates
*
* Input:
*   t            - Time at which orbit required (GPS time) J2000 sec
*   ae           - Equatorial radius of reference ellipsoid (unit)
*   flat         - Flattening of reference rellipsoid
*   torb[]       - Array of time tags (GPS time) J2000 sec
*   xorb[]       - Array of X coordinates (unit)
*   yorb[]       - Array of Y coordinates (unit)
*   zorb[]       - Array of Z coordinates (unit)
*   norb         - Number of point in orbit array
*   jpltxtfile[] - Name of JPL text file with orbit
*
* Output:
*   xyz[3]       - x,y,z coordinates (unit)
*   llh[3]       - Latitude (deg), Longitude (deg), height (unit)
*   xyzdot[3]    - x,y,z velocities (unit/sec)
* FUNCTION returns 1 is error interpolating orbit
**************************************************************************/
int interporb(double xyz[3], double llh[3], double xyzdot[3],
	      double t, double ae, double flat, double torb[],
	      double xorb[], double yorb[], double zorb[], int32_t norb)
{
   int32_t   i;

/* Interpolate x, y, z coordinates
*/
   for (i = 0; i < 3; ++i) {
     xyz[i] = 0.0;
     llh[i] = 0.0;
   }
   if (intlagrange(t, &xyz[0], norb, torb, xorb, 7, 1, &xyzdot[0]) != 0) {
     fprintf(stderr,"interporb: Error interpolating orbit: t = %f\n", t);
     return 1;
   }
   if (intlagrange(t, &xyz[1], norb, torb, yorb, 7, 1, &xyzdot[1]) != 0) {
     fprintf(stderr,"interporb: Error interpolating orbit: t = %f\n", t);
     return 1;
   }
   if (intlagrange(t, &xyz[2], norb, torb, zorb, 7, 1, &xyzdot[2]) != 0) {
     fprintf(stderr,"interporb: Error interpolating orbit: t = %f\n", t);
     return 1;
   }

   if (xyz2gd(&llh[0], &llh[1], &llh[2], xyz[0], xyz[1], xyz[2], ae, flat) == 1) {
     return 1;
   }

   return 0;
}
/**********************************************************************
* FUNCTION: XYZ2GD
* Purpose:
* To convert geocentric x,y,z coordinates into geodetic latitude,
* longitude and height
* Input:
*   x       - Geocentric X-coordinate (unit)
*   y       - Geocentric Y-coordinate (unit)
*   z       - Geocentric Z-coordinate (unit)
*   ae      - Radius of the Earth (unit)
*   f       - Flattening of the Earth  (1.0/298.257 for Earth)
*             if (flattening <= 0.0, then assumed that f = 0.0
*             and sphere is assumed (Geodetic quantities are then
*             equivalent to geocentric quantities))
* Output:
*   glat    - Geodetic latitude (deg) (<= +- 90.0 deg)
*   glon    - Geodetic longitude (deg) (>= 0 and < 360.0)
*   ght     - Geodetic height (unit)
**********************************************************************/
int xyz2gd(double *glat, double *glon, double *ght, double x,
	   double y, double z, double ae, double f)
{
   double r, p, lon, lat, h, oof, esq, x0, y0, hm, N, sn;
   double Nw2g, uw2g, vw2g, uw2h, vw2h, det, u, v, epsuv;
   int32_t nit;

   if (f < 0.0) {
     fprintf(stderr,"ERROR: xyz2gd - Flattening must be >= 0.0");
     return 1;
   }
   r     = sqrt(x*x + y*y + z*z);
   p     = sqrt(x*x + y*y);
   epsuv = ae*GEODET_EPSUV;

/* First deal with case at poles
*/
   if (p == 0.0) {
     if (z == 0.0) {
       fprintf(stderr,"ERROR: xyz2gd - Error all three components are equal to zero\n");
       return 1;
     }
     else {
       if (z > 0.0) {
	 lon = 0.0;
	 lat = PIO2;
	 h   = fabs(z) - ae*(1.0 - f);
       }
       else {
	 lon = 0.0;
	 lat = -PIO2;
	 h   = fabs(z) - ae*(1.0 - f);
       }
     }
   }
   else {
/* Compute longitude from 0 to 2pi
*/
     if (x == 0.0) {
       if (y > 0.0) {
	 lon = PIO2;
       }
       else {
	 lon = 3.0*PIO2; 
       }
     }
     else {
       lon = atan2(y, x);
       if (lon < 0.0) {
	 lon += TWOPI;
       }
     }
/* Compute geocentric latitude and height 
*/
     if (f <= 0.0) {
       lat = atan(z/p);
       h   = r - ae;
     }
     else {
/* Compute first approximation of geodetic latitude and height
*/
       oof = 1.0/f;
       esq = f*(2.0 - f);
       lat = atan((oof*oof*z)/(p*(1.0-oof)*(1.0-oof)));
       N   = ae/(sqrt(1.0 - esq*sin(lat)*sin(lat)));
       x0  = N*cos(lat);
       y0  = N*(1.0 - esq)*sin(lat);
       hm  = sqrt((p-x0)*(p-x0) + (z-y0)*(z-y0));
       sn  = r - sqrt(x0*x0 + y0*y0);
       if (sn < 0.0) {
	 h = -hm;
       }
       else {
	 h = hm;
       }

/* Iterate for geodetic latitude and height
*/
       u   = (N + h)*cos(lat) - p;
       v   = (N*(1.0 - esq) + h)*sin(lat) - z;
       nit = 0;
       while ((fabs(u) > epsuv) || (fabs(v) > epsuv)) {
	 if (nit > 100) {
	   fprintf(stderr,"ERROR: xyz2gd - Does not converge on latitude and height\n");
	   exit(1);
	 }

/* Partials
*/
	 Nw2g = (N/ae)*(N/ae)*N*esq*sin(lat)*cos(lat);
	 uw2g = Nw2g*cos(lat) - (N + h)*sin(lat);
	 vw2g = Nw2g*sin(lat)*(1.0 - esq) + (N*(1.0 - esq) + h)*cos(lat);
	 uw2h = cos(lat);
	 vw2h = sin(lat);

/* Corrections
*/
	 det  = uw2g*vw2h - uw2h*vw2g;
	 lat += (-vw2h*u + uw2h*v)/det;
	 h   += ( vw2g*u - uw2g*v)/det;

/* Error
*/
         N   = ae/(sqrt(1.0 - esq*sin(lat)*sin(lat)));
	 u   = (N + h)*cos(lat) - p;
	 v   = (N*(1.0 - esq) + h)*sin(lat) - z;
	 ++nit;
       }
     }
   }

/* Geodetic height
*  if (f > 0.0) {
*    esq = f*(2.0 - f);
*    num = z*(1.0 - f)*r + z*esq*ae;
*    den = p*r;
*    u   = atan2(num, den);
*    num = z*(1.0 - f)  + esq*ae*sin(u)*sin(u)*sin(u);
*    den = (1.0 - f)*(p - esq*ae*cos(u)*cos(u)*cos(u));
*    lat = atan2(num, den);
*    h   = p*cos(phi) + z*sin(phi) - ae*sqrt(1.0 - esq*sin(phi)*sin(phi));
*  }
*/

/* Convert to degrees
*/
   *glon = lon/DTR;
   *glat = lat/DTR;
   *ght  = h;

/* Checks on bounds
*/
   if ((*glat < -90.0) || (*glat > 90.0)) {
     fprintf(stderr,"ERROR: xyz2gd - Error with bounds of latitude\n");
     return 1;
   }
   if (*glon < 0.0) {
     *glon = *glon + 360.0;
   }

   return 0;
}
/******************************************************************************
*      RTG Source Code,                                                       *
*      Copyright (C) 1996, California Institute of Technology                 *
*      U.S. Government Sponsorship under NASA Contract NAS7-1260              *
*                    (as may be time to time amended)                         *
*                                                                             *
*      RTG is a trademark of the California Institute of Technology.          *
*                                                                             *
*                                                                             *
*      written by Yoaz Bar-Sever, Willy Bertiger, Bruce Haines,               *
*                 Angelyn Moore, Ron Muellerschoen, Tim Munson,               *
*                 Larry Romans, and Sien Wu                                   *
*                                                                             *
*      modified by Gerhard L.H. Kruizinga for stand alone use                 *
*                 7/22/98                                                     *
******************************************************************************/
/* Interpolates an ECI file to retrieve satellite state at the requested time*/
/* Yoaz Bar-Sever. May, 1996 */
#define MAX_DEG 20
/* Performs straightforward Lagrange interpolation,
  to get value y(x) and (if requested) derivative y'(x),
  given tables of x-y points xt[] and yt[].  Tables should
  be equally spaced in x; will still work otherwise, but
  search used is stupidest possible.

    ntab = size of tables
    ndeg = degree of polynomial (uses ndeg+1 points around target x,
           symmetrically distributed if possible)
*/
int intlagrange( double x, double *y, int32_t ntab, double *xt, double *yt, 
                  int32_t ndeg, int32_t compute_deriv, double *yd)
{

  double i_r, yyd;
  int32_t i_shift, i, i1, i2, j, k, n, n2;

  static double w[MAX_DEG], df[MAX_DEG], x0_save, x_save;
  double *xi, *yi;
  static int32_t n_save = -1;
  static int32_t i1_save;
 /*  double ddif[MAX_DEG][MAX_DEG]; */

  if (n_save == ndeg + 1 && x == x_save && xt[0] == x0_save
       && compute_deriv == 0) {
    *y = 0.0;
    yi = yt + i1_save;
    for (i = 0; i < n_save; ++i) {
      *y += w[i]*yi[i];
    }
    return (int) 0;
  }

  if (x < xt[0] ) { return (int) -1;}
  if (x > xt[ntab-1]) { return (int) 1; }
/*
    fprintf(stderr, "filup: time %.16g out of table range [ %.16g , %.16g ]\n",
      x, xt[0], xt[ntab-1]);
    exit(1);
*/


  if (ntab <= ndeg) {
/*
    fprintf(stderr, "filup: table size = %d, not big enough for degree = %d\n",
      ntab, ndeg);
    exit(1);
*/
    return (int) 2;
  }

  i_r = (ntab - 1) * (x - xt[0])/(xt[ntab-1] - xt[0]);
  i = (int32_t) floor(i_r);
  if (i == ntab - 1) i--;
  i_r = i_r - i;

  if (i == ntab-1) {
    i--;
    i_r++;
  }

  if (x < xt[i] || x > xt[i+1]) {
    i_shift = 0;
    if (x < xt[i]) {
      while (x < xt[i]) { i_shift--; i--; }
    }
    else {
      while (x > xt[i+1]) { i_shift++; i++; }
    }
/*
    EH(EH_warning, "filup: table apparently not evenly spaced;"
      " did dumb search to shift i by %d to %d.", i_shift, i);
*/
    i_r = (x - xt[i])/(xt[i+1] - xt[i]);
  }

  n = ndeg + 1;
  if (n % 2) {
    /* n odd */
    n2 = (n-1)/2;
    if (i_r < 0.5) {
      i1 = i - n2;
      i2 = i + n2;
    }
    else {
      i1 = i - n2 + 1;
      i2 = i + n2 + 1;
    }
  }
  else {
    n2 = n/2;
    i1 = i - n2 + 1;
    i2 = i + n2;
  }
  
  if (i1 < 0) {
    i1 = 0;
    i2 = n - 1;
  }
  if (i2 >= ntab) {
    i2 = ntab - 1;
    i1 = ntab - n;
  }

  i1_save = i1;
  xi = xt + i1;
  yi = yt + i1;

  for (i = 0; i < n; ++i) {
    df[i] = 1.0;
    for (j = 0; j < n; ++j) {
      if (j != i) {
        df[i] /= (xi[i] - xi[j]);
      }
    }
  }

  *y = 0.0;
  for (i = 0; i < n; ++i) {
    w[i] = df[i];
    for (j = 0; j < n; ++j) {
      if (j != i) {
        w[i] *= (x - xi[j]);
      }
    }
    *y += w[i]*yi[i];
  }

  if (compute_deriv == 1) {
    *yd = 0.0;
    for (i = i1; i <= i2; i++) {
      for (j = i1; j <= i2; j++) {
        if (j != i) {
          yyd = yt[i]/(xt[i] - xt[j]);
          for (k = i1; k <= i2; k++) {
            if (k != i && k != j) yyd *= (x - xt[k])/(xt[i] - xt[k]);
          }
          *yd += yyd;
        }
      }
    }
  }

  n_save = n;
  x0_save = xt[0];
  x_save = x;

  return (int) 0;
}

