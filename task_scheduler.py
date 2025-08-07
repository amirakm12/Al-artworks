from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
import logging
import time
from typing import Callable, Optional, Dict, Any, Union
from dataclasses import dataclass

logger = logging.getLogger("TaskScheduler")

@dataclass
class ScheduledJob:
    """Represents a scheduled job with metadata"""
    job_id: str
    func: Callable
    trigger: str
    next_run: Optional[time.time]
    args: tuple
    kwargs: dict
    misfire_grace_time: Optional[int]
    coalesce: bool
    max_instances: int

class TaskScheduler:
    """Advanced task scheduler with multiple job stores and trigger types"""
    
    def __init__(self, jobstore: str = "memory", timezone: str = "UTC"):
        self.jobstore = jobstore
        self.timezone = timezone
        self.jobs = {}
        
        # Configure job stores
        if jobstore == "memory":
            jobstores = {
                'default': MemoryJobStore()
            }
        elif jobstore == "sqlite":
            jobstores = {
                'default': SQLAlchemyJobStore(url='sqlite:///jobs.db')
            }
        else:
            jobstores = {
                'default': MemoryJobStore()
            }
        
        # Create scheduler
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            timezone=timezone,
            job_defaults={
                'coalesce': True,
                'max_instances': 3,
                'misfire_grace_time': 60
            }
        )
        
        # Start scheduler
        self.scheduler.start()
        logger.info(f"Task scheduler started with jobstore: {jobstore}")

    def add_cron_job(self, func: Callable, job_id: Optional[str] = None,
                     year: Optional[Union[int, str]] = None,
                     month: Optional[Union[int, str]] = None,
                     day: Optional[Union[int, str]] = None,
                     week: Optional[Union[int, str]] = None,
                     day_of_week: Optional[Union[int, str]] = None,
                     hour: Optional[Union[int, str]] = None,
                     minute: Optional[Union[int, str]] = None,
                     second: Optional[Union[int, str]] = None,
                     args: tuple = (), kwargs: dict = None,
                     misfire_grace_time: Optional[int] = None,
                     coalesce: bool = True,
                     max_instances: int = 1) -> str:
        """Add a cron-style scheduled job"""
        
        if kwargs is None:
            kwargs = {}
        
        trigger = CronTrigger(
            year=year, month=month, day=day, week=week,
            day_of_week=day_of_week, hour=hour, minute=minute, second=second,
            timezone=self.timezone
        )
        
        job = self.scheduler.add_job(
            func=func,
            trigger=trigger,
            id=job_id,
            args=args,
            kwargs=kwargs,
            misfire_grace_time=misfire_grace_time,
            coalesce=coalesce,
            max_instances=max_instances,
            replace_existing=True
        )
        
        logger.info(f"Added cron job: {job.id} @ {trigger}")
        return job.id

    def add_interval_job(self, func: Callable, job_id: Optional[str] = None,
                        weeks: int = 0, days: int = 0, hours: int = 0,
                        minutes: int = 0, seconds: int = 0,
                        args: tuple = (), kwargs: dict = None,
                        misfire_grace_time: Optional[int] = None,
                        coalesce: bool = True,
                        max_instances: int = 1) -> str:
        """Add an interval-based scheduled job"""
        
        if kwargs is None:
            kwargs = {}
        
        trigger = IntervalTrigger(
            weeks=weeks, days=days, hours=hours,
            minutes=minutes, seconds=seconds,
            timezone=self.timezone
        )
        
        job = self.scheduler.add_job(
            func=func,
            trigger=trigger,
            id=job_id,
            args=args,
            kwargs=kwargs,
            misfire_grace_time=misfire_grace_time,
            coalesce=coalesce,
            max_instances=max_instances,
            replace_existing=True
        )
        
        logger.info(f"Added interval job: {job.id} @ {trigger}")
        return job.id

    def add_date_job(self, func: Callable, run_date: Union[str, time.time],
                    job_id: Optional[str] = None,
                    args: tuple = (), kwargs: dict = None,
                    misfire_grace_time: Optional[int] = None) -> str:
        """Add a one-time scheduled job"""
        
        if kwargs is None:
            kwargs = {}
        
        trigger = DateTrigger(
            run_date=run_date,
            timezone=self.timezone
        )
        
        job = self.scheduler.add_job(
            func=func,
            trigger=trigger,
            id=job_id,
            args=args,
            kwargs=kwargs,
            misfire_grace_time=misfire_grace_time,
            replace_existing=True
        )
        
        logger.info(f"Added date job: {job.id} @ {trigger}")
        return job.id

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job"""
        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Removed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove job {job_id}: {e}")
            return False

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get job information"""
        job = self.scheduler.get_job(job_id)
        if job:
            return ScheduledJob(
                job_id=job.id,
                func=job.func,
                trigger=str(job.trigger),
                next_run=job.next_run_time.timestamp() if job.next_run_time else None,
                args=job.args,
                kwargs=job.kwargs,
                misfire_grace_time=job.misfire_grace_time,
                coalesce=job.coalesce,
                max_instances=job.max_instances
            )
        return None

    def get_jobs(self) -> list[ScheduledJob]:
        """Get all scheduled jobs"""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append(ScheduledJob(
                job_id=job.id,
                func=job.func,
                trigger=str(job.trigger),
                next_run=job.next_run_time.timestamp() if job.next_run_time else None,
                args=job.args,
                kwargs=job.kwargs,
                misfire_grace_time=job.misfire_grace_time,
                coalesce=job.coalesce,
                max_instances=job.max_instances
            ))
        return jobs

    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job"""
        try:
            self.scheduler.pause_job(job_id)
            logger.info(f"Paused job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        try:
            self.scheduler.resume_job(job_id)
            logger.info(f"Resumed job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False

    def modify_job(self, job_id: str, **kwargs) -> bool:
        """Modify an existing job"""
        try:
            self.scheduler.modify_job(job_id, **kwargs)
            logger.info(f"Modified job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to modify job {job_id}: {e}")
            return False

    def get_job_count(self) -> int:
        """Get total number of scheduled jobs"""
        return len(self.scheduler.get_jobs())

    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler"""
        self.scheduler.shutdown(wait=wait)
        logger.info("Task scheduler shut down")

    def pause(self):
        """Pause all jobs"""
        self.scheduler.pause()
        logger.info("All jobs paused")

    def resume(self):
        """Resume all jobs"""
        self.scheduler.resume()
        logger.info("All jobs resumed")

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        jobs = self.get_jobs()
        running_jobs = [j for j in jobs if j.next_run and j.next_run > time.time()]
        paused_jobs = [j for j in jobs if not j.next_run or j.next_run <= time.time()]
        
        return {
            "total_jobs": len(jobs),
            "running_jobs": len(running_jobs),
            "paused_jobs": len(paused_jobs),
            "scheduler_running": self.scheduler.running,
            "next_job_time": min([j.next_run for j in running_jobs]) if running_jobs else None
        }

# Example usage functions
def example_daily_task():
    """Example daily task"""
    logger.info("Running daily task...")
    # Add your daily task logic here

def example_hourly_task():
    """Example hourly task"""
    logger.info("Running hourly task...")
    # Add your hourly task logic here

def example_backup_task():
    """Example backup task"""
    logger.info("Running backup task...")
    # Add your backup logic here

def example_cleanup_task():
    """Example cleanup task"""
    logger.info("Running cleanup task...")
    # Add your cleanup logic here

# Example usage
def example_scheduler():
    """Example of using the task scheduler"""
    logging.basicConfig(level=logging.INFO)
    
    # Create scheduler
    scheduler = TaskScheduler(jobstore="memory")
    
    try:
        # Add various types of jobs
        scheduler.add_cron_job(
            example_daily_task,
            job_id="daily_task",
            hour=8,
            minute=0,
            args=("Daily task",)
        )
        
        scheduler.add_cron_job(
            example_hourly_task,
            job_id="hourly_task",
            minute=0,
            second=0
        )
        
        scheduler.add_interval_job(
            example_backup_task,
            job_id="backup_task",
            hours=6
        )
        
        scheduler.add_date_job(
            example_cleanup_task,
            job_id="cleanup_task",
            run_date="2024-01-01 00:00:00"
        )
        
        # Get job information
        jobs = scheduler.get_jobs()
        logger.info(f"Scheduled {len(jobs)} jobs:")
        for job in jobs:
            logger.info(f"  {job.job_id}: {job.trigger}")
        
        # Get stats
        stats = scheduler.get_stats()
        logger.info(f"Scheduler stats: {stats}")
        
        # Keep running for a while
        import time
        time.sleep(10)
        
    finally:
        scheduler.shutdown()

if __name__ == "__main__":
    example_scheduler()