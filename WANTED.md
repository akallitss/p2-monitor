# p2basket-monitoring

## Generic format

time; Quantity_1; Value_1; ... ; Quantity_i; Value_i; ... ; Quantity_N ; Value_N

i = 1, ..., N : channel number

## Monitoring

- quantities : Rates; <ADC>; ...
- Configuration : Time interval; Back-end IPs; VMM indices (0-15, in back-end coordinates)
- option for a CSV file, live plot, or plot to file

## Data

- Basic analysis : hit rates per channel, per VMM, or per Hybrid?
- hit rates per FMC (as they cover different detector planes)
- per back-end : total hit rate, data rate
- hit rates per detector module, detector plane, grand total

Lines 2,3,4 calculated from the appropriate sums from line 1