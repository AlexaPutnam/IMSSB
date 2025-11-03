<<comment
for cyc in $(seq 301 315)
do
        rads2nc -X imssb_rads.xml -S sw -C $cyc -V time,pass_number,lat,lon,alt,range_ku,range_c,range_ku_mle3,swh_ku,swh_c,swh_ku_mle3,swh_rms_ku,swh_rms_c,swh_rms_ku_mle3,sig0_ku,sig0_c,sig0_ku_mle3,sig0_rms_ku,sig0_rms_c,sig0_rms_ku_mle3,range_rms_ku,range_rms_c,range_rms_ku_mle3,rad_wet_tropo_corr,model_wet_tropo_corr,model_dry_tropo_corr,iono_corr_alt_ku,iono_corr_alt_ku_mle3,off_nadir_angle2_wf_rms_ku,pole_tide,ocean_tide_non_eq,internal_tide_hret,dac,sea_state_bias_ku,sea_state_bias_c,sea_state_bias_mle3,iono_corr_gim_ku,ocean_tide_fes,ocean_tide_got,load_tide_fes,load_tide_got,ssha,ssha_mle3,mean_sea_surface,solid_earth_tide -o /home/aputnam/IMSSB/imssb_software/output/sw/product2data/rads/sw_dir_product2data_rads_"$cyc"_c_"$cyc"_na_na.nc -v
done
comment
<<comment
for cyc in $(seq 99 129)
do
        rads2nc -X imssb_rads.xml -S 6a -C $cyc -V time,pass_number,lat,lon,alt,range_ku,range_c,range_ku_mle3,swh_ku,swh_c,swh_ku_mle3,swh_rms_ku,swh_rms_c,swh_rms_ku_mle3,sig0_ku,sig0_c,sig0_ku_mle3,sig0_rms_ku,sig0_rms_c,sig0_rms_ku_mle3,range_rms_ku,range_rms_c,range_rms_ku_mle3,rad_wet_tropo_corr,model_wet_tropo_corr,model_dry_tropo_corr,iono_corr_alt_ku,iono_corr_alt_ku_mle3,off_nadir_angle2_wf_rms_ku,pole_tide,ocean_tide_non_eq,internal_tide_hret,dac,sea_state_bias_ku,sea_state_bias_c,sea_state_bias_mle3,iono_corr_gim_ku,ocean_tide_fes,ocean_tide_got,load_tide_fes,load_tide_got,ssha,ssha_mle3,mean_sea_surface,solid_earth_tide -o /home/aputnam/IMSSB/imssb_software/output/6a/product2data/rads/6a_dir_product2data_rads_"$cyc"_c_"$cyc"_na_na.nc -v
done
comment
<<comment
for cyc in $(seq 102 278)
do
        rads2nc -X imssb_rads.xml -S sw -C $cyc -V time,pass_number,lat,lon,alt,range_ku,range_c,range_ku_mle3,swh_ku,swh_c,swh_ku_mle3,swh_rms_ku,swh_rms_c,swh_rms_ku_mle3,sig0_ku,sig0_c,sig0_ku_mle3,sig0_rms_ku,sig0_rms_c,sig0_rms_ku_mle3,range_rms_ku,range_rms_c,range_rms_ku_mle3,rad_wet_tropo_corr,model_wet_tropo_corr,model_dry_tropo_corr,iono_corr_alt_ku,iono_corr_alt_ku_mle3,off_nadir_angle2_wf_rms_ku,pole_tide,ocean_tide_non_eq,internal_tide_hret,dac,sea_state_bias_ku,sea_state_bias_c,sea_state_bias_mle3,iono_corr_gim_ku,ocean_tide_fes,ocean_tide_got,load_tide_fes,load_tide_got,ssha,ssha_mle3,mean_sea_surface,solid_earth_tide,wind_speed_alt,dist_coast,topo,surface_type_rad1,surface_type_rad2 -o /home/aputnam/IMSSB/imssb_software/output/sw/product2data/rads/sw_dir_product2data_rads_"$cyc"_c_"$cyc"_na_na.nc -v
done
comment
for cyc in $(seq 102 278)
do
        rads2nc -X imssb_rads.xml -S sw -C $cyc -V time,pass_number,lat,lon,alt,range_ku,range_c,swh_ku,swh_c,swh_rms_ku,swh_rms_c,sig0_ku,sig0_c,sig0_rms_ku,sig0_rms_c,range_rms_ku,range_rms_c,rad_wet_tropo_corr,model_wet_tropo_corr,model_dry_tropo_corr,iono_corr_alt_ku,off_nadir_angle2_wf_rms_ku,pole_tide,ocean_tide_non_eq,internal_tide_hret,dac,sea_state_bias_ku,sea_state_bias_c,iono_corr_gim_ku,ocean_tide_fes,ocean_tide_got,load_tide_fes,load_tide_got,ssha,ssha_mle3,mean_sea_surface,solid_earth_tide,wind_speed_alt,dist_coast,topo,surface_type_rad1,surface_type_rad2 -o /home/aputnam/IMSSB/imssb_software/output/sw/product2data/rads/sw_dir_product2data_rads_"$cyc"_c_"$cyc"_na_na.nc -v
done
