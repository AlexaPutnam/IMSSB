
for cyc in $(seq 36 111)
do
        rads2nc -X imssb_rads.xml -S sa -C $cyc -V time,pass_number,lat,lon,alt,range_ka,swh_ka,swh_rms_ka,sig0_ka,sig0_rms_ka,range_rms_ka,rad_wet_tropo_corr,model_wet_tropo_corr,model_dry_tropo_corr,pole_tide,ocean_tide_non_eq,internal_tide_hret,dac,ssb_tran2019,iono_gim,iono_nic09,ocean_tide_fes,ocean_tide_got,load_tide_fes,load_tide_got,ssha_ka,mean_sea_surface,solid_earth_tide,topo_srtm30plus,dist_coast,wind_speed_alt,mean_sea_surface_cnes15 -o /home/aputnam/IMSSB/imssb_software/output/sa/product2data/rads/radsout/sa_dir_product2data_rads_"$cyc"_c_"$cyc"_na_na.nc -v
done
