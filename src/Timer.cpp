#include "Timer.h"

int
Timer::start()
{
    this->old_wc_time = glu_monotonic_now();
    if (getrusage (RUSAGE_SELF, &this->old_us_time) == -1)
        return -1;
    else
        return 0;
}

int
Timer::elapsedWallclockTime (double &wc)
{
    // Preserve the existing public API's millisecond unit.
    wc = 1000.0 * glu_seconds_between(this->old_wc_time, glu_monotonic_now());
    return 0;
}

int
Timer::elapsedUserTime (double &ut)
{
    if (getrusage (RUSAGE_SELF, &this->new_us_time) == -1)
        return -1;
    ut = (this->new_us_time.ru_utime.tv_sec - this->old_us_time.ru_utime.tv_sec) * 1000.0
	        + ((this->new_us_time.ru_utime.tv_usec
            - this->old_us_time.ru_utime.tv_usec) / 1000.0);
    return 0;
}

int
Timer::elapsedSystemTime (double &st)
{
    if (getrusage (RUSAGE_SELF, &this->new_us_time) == -1)
        return -1;
    st = (this->new_us_time.ru_stime.tv_sec - this->old_us_time.ru_stime.tv_sec) * 1000.0
	        + ((this->new_us_time.ru_stime.tv_usec
            - this->old_us_time.ru_stime.tv_usec) / 1000.0);
    return 0;
}

int
Timer::elapsedTime (double &wallclock, double &user_time, double &system_time)
{
    if (this->elapsedWallclockTime (wallclock) == -1)
        return -1;
    else
    {
        if (getrusage (RUSAGE_SELF, &this->new_us_time) == -1)
	        return -1;
        user_time = (this->new_us_time.ru_utime.tv_sec
		                - this->old_us_time.ru_utime.tv_sec) * 1000.0
				        + ((this->new_us_time.ru_utime.tv_usec
			            - this->old_us_time.ru_utime.tv_usec) / 1000.0);
        system_time = (this->new_us_time.ru_stime.tv_sec
		                - this->old_us_time.ru_stime.tv_sec) * 1000.0
				        + ((this->new_us_time.ru_stime.tv_usec
			            - this->old_us_time.ru_stime.tv_usec) / 1000.0);
        return 0;
    }
}
