import asyncio
from typing import TypeVar, Generic, Callable, Awaitable, Optional
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a generic type variable for the data being cached
T = TypeVar('T')

class AsyncCachedGenerator(Generic[T]):
    """
    A generic class to cache the result of an async data generation function.

    It provides an initial value quickly from the cache if available, 
    while triggering a background update for subsequent requests.
    """
    def __init__(self, generator_func: Callable[[], Awaitable[T]], name: str = "Generic"):
        """
        Initializes the cache handler.

        Args:
            generator_func: An asynchronous function (coroutine) that takes no arguments 
                            and returns the data to be cached.
            name: An optional name for logging purposes to identify the cache instance.
        """
        if not asyncio.iscoroutinefunction(generator_func):
            raise TypeError("generator_func must be an async function (coroutine).")
            
        self._generator_func = generator_func
        self._cached_data: Optional[T] = None
        self._update_lock = asyncio.Lock()
        self._initial_generation_complete = asyncio.Event()
        self._name = name # For logging

    async def _perform_generation(self) -> T:
        """Calls the provided generator function."""
        logger.info(f"[{self._name}] Performing data generation...")
        try:
            data = await self._generator_func()
            logger.info(f"[{self._name}] Data generation complete.")
            return data
        except Exception as e:
            logger.error(f"[{self._name}] Error during data generation: {e}", exc_info=True)
            # Decide error handling: re-raise, return default, return old cache?
            # For now, re-raise to make the caller aware.
            raise 

    async def _update_cache_background(self):
        """Task to run in the background to update the cache."""
        if self._update_lock.locked():
            logger.info(f"[{self._name}] Update already in progress, skipping new background task.")
            return
        
        async with self._update_lock:
            logger.info(f"[{self._name}] Starting background cache update.")
            try:
                new_data = await self._perform_generation()
                self._cached_data = new_data
                self._initial_generation_complete.set() # Mark initial generation as done
                logger.info(f"[{self._name}] Cache updated successfully in background.")
            except Exception as e:
                # Log error but don't kill the background task if possible
                logger.error(f"[{self._name}] Error during background cache update: {e}", exc_info=True)
            # Lock is released automatically

    async def get_data(self) -> T:
        """
        Gets the data. Returns cached data immediately if available, 
        otherwise waits for the initial generation. Triggers background 
        updates for subsequent requests.
        """
        if self._cached_data is not None:
            logger.info(f"[{self._name}] Cache hit. Returning cached data.")
            # Trigger background update if no update is currently running
            if not self._update_lock.locked():
                logger.info(f"[{self._name}] Triggering background cache update.")
                asyncio.create_task(self._update_cache_background())
            else:
                 logger.info(f"[{self._name}] Background update already in progress.")
            return self._cached_data
        else:
            # Cache miss: Need to wait for the first generation
            logger.info(f"[{self._name}] Cache miss. Waiting for initial generation.")
            # Use lock to ensure only one initial generation happens
            async with self._update_lock:
                # Double-check if cache was populated while waiting for the lock
                if self._cached_data is not None:
                    logger.info(f"[{self._name}] Cache populated while waiting for lock. Returning cached data.")
                    # Optionally trigger background update here too if logic requires it
                    return self._cached_data

                logger.info(f"[{self._name}] Performing initial generation.")
                try:
                    initial_data = await self._perform_generation()
                    self._cached_data = initial_data
                    self._initial_generation_complete.set() # Signal that initial generation is done
                    logger.info(f"[{self._name}] Initial generation complete. Cache populated.")
                    return initial_data
                except Exception as e:
                    # If initial generation fails, we don't have anything to return.
                    # The lock will be released, subsequent calls might retry.
                    logger.error(f"[{self._name}] Initial generation failed: {e}", exc_info=True)
                    raise # Re-raise the exception
            
    async def wait_for_initial_generation(self):
        """Waits until the first successful generation has completed."""
        await self._initial_generation_complete.wait()

    def is_ready(self) -> bool:
        """Checks if the initial generation has completed."""
        return self._initial_generation_complete.is_set()
        
    def get_cached_data_sync(self) -> Optional[T]:
        """Synchronously returns the current cached data, if any."""
        return self._cached_data 