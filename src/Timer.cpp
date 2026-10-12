#include "Timer.h"

int
Timer::start()
{
    this->old_wc_time = std::chrono::steady_clock::now();
    if (getrusage (RUSAGE_SELF, &this->old_us_time) == -1)
        return -1;
    else
        return 0;
}

int
Timer::elapsedWallclockTime (double &wc)
{
    wc = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - this->old_wc_time).count();
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
