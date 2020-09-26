File names:

    <prefix>_<side>_<interval>_3months.csv

<side>:     length (meters) of the square'side
<interval>: length (minutes) of the time interval
            when 'interval = 0' its length is 5 seconds.

Start date:  (2020-01-01 0:0:0)
End date:    (2020-04-01 0:0:0)    excluded, or
             (2020-03-31 23:59:59) included


How to convert a coordinate (i,j,t) in (latitude, longitude, date, time)

i -> latitude  = i*111319/side
j -> longitude = j*111319/side
t -> (date, time) = date(2020-01-01 0:0:0) + time_duration(minutes=t*interval)

Remember that 'time_duration' is a 'duration measure' with 'measure unit' (seconds, minutes, hours, ...)
