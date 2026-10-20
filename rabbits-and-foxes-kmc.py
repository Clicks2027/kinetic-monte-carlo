# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: .venv (3.14.7)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Rabbits and foxes, by kinetic Monte Carlo
#
# **CHME 5137, Lecture 10, Tuesday October 13, 2026.** Started in class, finished for
# homework.
#
# *Written for 2026 by Claude Code (Claude Opus 5.5) from Prof. West's KMC question of
# earlier years, and checked by Prof. West. The problem is from Chapter 1 of H. Scott
# Fogler's "Essentials of Chemical Reaction Engineering".*
#
# ---
#
# Same farm, same rate constants: 400 rabbits and 200 foxes at the start, and
#
# $$\frac{dR}{dt} = k_1 R - k_2 R F \qquad\qquad \frac{dF}{dt} = k_3 R F - k_4 F$$
#
# with $k_1 = 0.015$ day$^{-1}$, $k_2 = 0.00004$ day$^{-1}$ foxes$^{-1}$,
# $k_3 = 0.0004$ day$^{-1}$ rabbits$^{-1}$ and $k_4 = 0.04$ day$^{-1}$.
#
# In the `solve_ivp` homework you found that the foxes fall to **under two** in each
# trough. The ODEs don't mind: 1.6 foxes carry on and breed. Real foxes can't be
# fractional, and two foxes can easily both die before they meet a rabbit. So now we
# treat every rabbit and fox as an individual, and every birth and death as a random
# event. That is **kinetic Monte Carlo** (KMC).
#
# ### The events
#
# There are four events. Each has a rate, in events per day, which changes as the
# populations change:
#
# | Event | Rate | Effect |
# |---|---|---|
# | a rabbit is born | $k_1 R$ | $R \to R + 1$ |
# | a rabbit is eaten | $k_2 R F$ | $R \to R - 1$ |
# | a fox is born | $k_3 R F$ | $F \to F + 1$ |
# | a fox dies | $k_4 F$ | $F \to F - 1$ |
#
# The rates are easy to calculate, so we use the **rejection-free** algorithm, which you
# worked out in class from
# [Wikipedia's Kinetic Monte Carlo page](https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo).
# Writing it down in your own words is the first part of task 1.
#
# ### How to work
#
# - **Commit as you go.** After each numbered task works, Restart & Run All, then
#   stage, commit with a message that says what you did, and push. At least one commit
#   per task. Your commit history is part of what gets looked at.
# - **Answer the questions in words**, in the markdown cells marked *Answer*.
#   Double-click a cell to edit it.
# - **Random numbers:** use a NumPy generator, `rng = np.random.default_rng(seed)`,
#   not the legacy `np.random.*` functions. Give it a seed, so that Restart & Run All
#   gives the same answers every time. Check what interval each function returns,
#   and what its arguments mean, before you trust it.
# - **Let bugs shout.** If a population goes negative or an event isn't one of the
#   four, `raise` an error. Don't hide it with `max(0, F)`: that turns a bug you'd
#   notice into a wrong answer you won't.

# %%
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

rng = np.random.default_rng(5137)

k1 = 0.015  # 1/day
k2 = 0.00004  # 1/day/fox
k3 = 0.0004  # 1/day/rabbit
k4 = 0.04  # 1/day

R0, F0 = 400, 200  # rabbits and foxes at t = 0
t_end = 600  # days

# %% [markdown]
# The deterministic answer, for comparison. This is the `solve_ivp` homework's
# solution, at a tight tolerance.

# %%
def dydt(t, y):
    R, F = y
    return [k1 * R - k2 * R * F, k3 * R * F - k4 * F]

ode = solve_ivp(dydt, (0, t_end), [R0, F0], rtol=1e-10, atol=1e-8, dense_output=True)
t_ode = np.linspace(0, t_end, 3001)
R_ode, F_ode = ode.sol(t_ode)

plt.plot(t_ode, R_ode, label="rabbits (ODE)")
plt.plot(t_ode, F_ode, label="foxes (ODE)")
plt.xlabel("time (days)")
plt.ylabel("population")
plt.legend()
plt.show()


# %% [markdown]
# ## 1. One trajectory
#
# **(a) The algorithm, in your own words.** Before you write any code, write the
# rejection-free algorithm as numbered steps, for rabbits and foxes: what each of
# Wikipedia's symbols ($k$, $r_{ki}$, $R_{ki}$, $Q_k$, $u$, $u'$) becomes here, how the
# next event is chosen, and how far the clock moves. Then work the **first step by
# hand**, from 400 rabbits and 200 foxes, with $u = 0.5$ and $u' = 0.3$: which event
# happens, and how many minutes pass?

# %%

# %%

# %% [markdown]
# Rejection free algorithim as numbered steps:
# 1. set t = 0
# 2. define variables at time = 0
# 3. create list of events in system
# 4. sum the events and calculate the total rate which is the sum of every event
# 5. generate a random uniform number
# 6. use this number to find the probability of event occuring based on its rate
# 7. update the algorithim to include the current state as the chosen event
# 8. generate another uniform random number
# 9. use this random number to calculate the change in time between previous event and new event 
# 10. repeat process at step 3 
#
#
#
# What Wikipedia's symbols become
# k becomes intial population of rabbits and foxes F_0 and R_0 
# r_ki becomes the product of the rate constants and varible. For example k_1 * R. R_ki becomes the sum of transition rates through event i. Q_k is the total rate of all events i or (rabbit birth) through N_k (fox death). u would be used in the context of the rabits and foxes problem to randomly select an event that occurs based on the event's rate. u' is also a random number generated to determine how long until the next event of any type happens. 
#
# Based on paper calculations, u*Q_k yielded 24.6 which tells us that fox birth happens. 
# 35.2382 minutes pass until the next event
#
# Number of events: 14256
# Runtime: 0.111996 seconds
#
# The KMC trajectory differs from the ODE solution in that the fox population dies in the KMC trajectory at 150 days and the rabbit population thrives after that (drastically rises). The ODE solution shows distinct peaks for the foxes and rabbits. 
#
#

# %%

# %% [markdown]
# **(b) The function.** Write
#
# ```python
# def kmc(R0, F0, t_end, rng):
#     ...
#     return times, rabbits, foxes   # three NumPy arrays, one entry per event
# ```
#
# that runs your algorithm from $t = 0$ until the clock passes `t_end`. If the total
# rate is ever zero, nothing more can happen, so stop there.
#
# Run it once. How many events were there, and how long did it take? Plot rabbits and
# foxes against time, on top of the ODE solution. (`plt.step(times, foxes, where="post")`
# draws a population that jumps, rather than one that slopes.) Run it a few more times
# with different seeds. How does a KMC trajectory differ from the ODE solution?

# %%
def kmc(R0, F0, t_end, rng):
    t = 0.0
    R, F = R0, F0

    times = [t]
    rabbits_events = [R]
    foxes_events = [F]

    while t < t_end:
        rates = np.array([k1*R, k2*R*F, k3*R*F, k4*F])
        total_rate = rates.sum()

        if total_rate == 0:
            break
        cumulative = np.cumsum(rates)
        u1 = rng.random()
        event = np.searchsorted(cumulative, u1 * total_rate)

        delta_t = 1/total_rate * np.log(1/rng.random())

        if event == 0:
            R = R +1
        elif event == 1:
            R = R - 1
        elif event == 2:
            F = F + 1
        else: 
            F = F - 1

        t = t + delta_t

        times.append(t)
        rabbits_events.append(R)
        foxes_events.append(F)



    



    return np.array(times), np.array(rabbits_events), np.array(foxes_events)    # three NumPy arrays, one entry per event
 

# %%
rng = np.random.default_rng(seed=1000)

start = time.perf_counter()
times, rabbits_events, foxes_events = kmc(R0, F0, t_end, rng)
elapsed = time.perf_counter() - start

print("Number of events:", len(times) - 1)
print(f"Runtime: {elapsed:.6f} seconds")


plt.plot(t_ode, R_ode, "--", color="tab:blue", label="Rabbits (ODE)")
plt.plot(t_ode, F_ode, "--", color="tab:orange", label="Foxes (ODE)")

plt.step(times, rabbits_events, where="post", color="tab:blue", label="Rabbits (KMC)")
plt.step(times, foxes_events, where="post", color="tab:orange", label="Foxes (KMC)")

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.show()

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## 2. Many trajectories, and where they end up
#
# Plot the foxes from 20 trajectories on one graph (thin lines, `lw=0.5`), with the ODE
# solution on top.
#
# - What do the trajectories have in common, and where do they spread apart?
# - What happens to a trajectory once the foxes reach zero? Once the rabbits reach
#   zero? Why can't it ever recover?
# - Where in the cycle do the foxes usually die out, and why there? (Your answer to
#   task 5 of the `solve_ivp` homework is relevant.)

# %%
plt.figure()
for i in range(20):
    rng = np.random.default_rng(seed=1000 + i)
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)
    plt.step(times, foxes, where="post", lw=0.5, color = "tab:orange", label = "Foxes (KMC)" if i == 0 else None)
plt.plot(t_ode, F_ode, "--", color="tab:blue", label="Foxes (ODE)")
plt.xlabel("time (days)")
plt.ylabel("foxes")
plt.legend()
plt.show()   
    

# %%

# %%

# %% [markdown]
# The trajectories all have a similar peak population, they shift slightly in time (x-axis). One fox population hits 0 early on and never recovers. The fox population plateus when it hits 0, this makes logical sense because once there are no foxes left, they cannot give birth to new ones and multiply. The foxes most often die out around 500 days, this is after the most often fox peak at 400ish days, where the rabbit population is at a minimum because they have been eaten by the foxes. With the rabbit population at a minimum, the foxes begin to perish.

# %% [markdown]
# ## 3. The second fox peak
#
# The ODE's foxes peak at about day 32, and again at about day 428, at about 2804
# foxes each time. When is the second peak in a KMC trajectory, and how high?
#
# A noisy trajectory has lots of tiny local maxima, so we need a definition. Use this
# one: **the largest number of foxes after day 200, counted as a peak only if it is
# above 100 foxes and isn't within 5 days of either end of the window** (day 200 or
# day 600). Write a function `second_peak(times, foxes)` that returns `(time, height)`,
# or `None` if there isn't one.
#
# Run 100 trajectories. In what fraction is there a second peak? For those that have
# one, give the mean and median of its time and of its height, and the interquartile
# range (25th to 75th percentile, `np.percentile`) of each. Plot histograms.
#
# How do these compare with the ODE's second peak? If they differ, why? And what does
# the definition get wrong? Find a trajectory where it picks something you wouldn't
# call "the second peak", if you can.

# %%
def second_peak(times, foxes):
    valid_times = (times > 205) & (times < 595)

    if not np.any(valid_times):
        return None

    candidates = np.flatnonzero(valid_times)
    peak_index = candidates[np.argmax(foxes[valid_times])]
    peak_height = foxes[peak_index]

    if peak_height <= 100:
        return None

    return times[peak_index], peak_height
peaks = []

for i in range(100):
    rng = np.random.default_rng(1000 + i)
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)

    peak = second_peak(times, foxes)
    if peak is not None:
        peaks.append(peak)

peaks = np.array(peaks)

print(f"Fraction with a second peak: {len(peaks) / 100:.2f}")

if len(peaks) > 0:
    peak_times = peaks[:, 0]
    peak_heights = peaks[:, 1]

    for name, values in [("Time", peak_times), ("Height", peak_heights)]:
        q25, q75 = np.percentile(values, [25, 75])
        print(
            f"{name}: mean={np.mean(values):.1f}, "
            f"median={np.median(values):.1f}, "
            f"IQR=({q25:.1f}, {q75:.1f})"
        )

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(peak_times)
    axes[0].set(xlabel="second-peak time (days)", ylabel="count")
    axes[1].hist(peak_heights)
    axes[1].set(xlabel="second-peak height (foxes)", ylabel="count")
    plt.tight_layout()
    plt.show()



# %% [markdown]
# Fraction with a second peak: 0.28
# Time: mean=398.7, median=391.0, IQR=(383.2, 416.7)
# Height: mean=2326.0, median=2262.0, IQR=(1997.0, 2613.5)
# ODE has second peak at 428 days of 2804 foxes
# They differ in that KMC returns a second peak sooner than ODE of drastically less foxes. They differ because ODE averages population whereas KMC includes randomness in birth and death of populations. The definition is wrong in that it arbitrarily decides that the second peak has to be beyond 200 days and above 100 foxes. It is also prone to calling rapid fluctuations a second peak

# %% [markdown]
# ## 4. How many runs are enough?
#
# Every number in task 3 came from random runs, so it has its own uncertainty. Repeat
# task 3 with 100, 300, 1000 and 3000 trajectories. (Time it first, and work out how
# long 3000 will take before you start.)
#
# - How do the mean second-peak time and height change with the number of runs?
# - Work out the **standard error** of each mean, $s / \sqrt{n}$, where $s$ is the
#   standard deviation and $n$ the number of trajectories with a peak. How does it
#   scale with $n$?
# - Give the expected time and height of the second peak, and their interquartile
#   ranges, to a precision you can justify. Say how you justified it.

# %%
import time

all_peaks = []
start = time.perf_counter()

for i in range(100):
    rng = np.random.default_rng(1000 + i)
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)
    all_peaks.append(second_peak(times, foxes))

elapsed_100 = time.perf_counter() - start
estimated_3000 = elapsed_100 * 30

print(f"100 trajectories took {elapsed_100:.1f} seconds.")
print(
    f"Estimated time for 3000 trajectories: "
    f"{estimated_3000:.1f} seconds")

# %%
start_remaining = time.perf_counter()

for i in range(100, 3000):
    rng = np.random.default_rng(1000 + i)
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)
    all_peaks.append(second_peak(times, foxes))

elapsed_remaining = time.perf_counter() - start_remaining
print(f"Remaining 2900 trajectories took {elapsed_remaining:.1f} seconds.")

# %%
for N in [100, 300, 1000, 3000]:
    peaks_N = [peak for peak in all_peaks[:N] if peak is not None]

    print(f"\nFirst {N} trajectories: {len(peaks_N)} had a second peak")
    print(f"Fraction with a second peak: {len(peaks_N) / N:.3f}")

    if len(peaks_N) == 0:
        continue

    peak_array = np.array(peaks_N)
    peak_times = peak_array[:, 0]
    peak_heights = peak_array[:, 1]

    for name, values in [
        ("Time (days)", peak_times),
        ("Height (foxes)", peak_heights),
    ]:
        q25, q75 = np.percentile(values, [25, 75])
        mean = np.mean(values)
        median = np.median(values)

        if len(values) >= 2:
            standard_error = np.std(values, ddof=1) / np.sqrt(len(values))
            se_text = f"{standard_error:.1f}"
        else:
            se_text = "not available (fewer than 2 peaks)"

        print(
            f"{name}: mean={mean:.1f}, median={median:.1f}, "
            f"IQR=({q25:.1f}, {q75:.1f}), SE={se_text}")

peaks_3000 = [peak for peak in all_peaks if peak is not None]

if peaks_3000:
    peak_array = np.array(peaks_3000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(peak_array[:, 0])
    axes[0].set(xlabel="second-peak time (days)", ylabel="count")
    axes[1].hist(peak_array[:, 1])
    axes[1].set(xlabel="second-peak height (foxes)", ylabel="count")
    plt.tight_layout()
    plt.show()
    

# %% [markdown]
# The mean second peak time --> increases with n
# The mean second peak height --> increases with n until 1000 trajectories, then decreases
# standard error decreases with increasing n
# Of most trajectories (3000), 30.9% had a second peak, this peak was at mean time 402.2 days at a mean peak height of 2386 foxes, IQR=(382.5, 418.8). This data had the lowest standard error. 

# %% [markdown]
# ## 5. How likely is it that the foxes die out?
#
# From your largest set of runs, what fraction have **no foxes left at day 600**?
# That fraction $p$ estimates a probability. Its standard error is
# $\sqrt{p(1-p)/N}$ for $N$ runs.
#
# - Give the probability that the foxes die out before 600 days, with its uncertainty.
# - How many runs would you need to know it to within ±1 percentage point? To within
#   ±0.1?
# - Plot a histogram of the times at which the foxes die out. When does it usually
#   happen?

# %%
N = 3000
extinction_times = np.full(N, np.nan)

for i in range(N):
    rng = np.random.default_rng(1000 + i)
    times, rabbits, foxes = kmc(R0, F0, t_end, rng)

    extinct_indices = np.flatnonzero((times <= 600) & (foxes == 0))

    if len(extinct_indices) > 0:
        extinction_times[i] = times[extinct_indices[0]]

extinct = np.isfinite(extinction_times)
p = np.mean(extinct)
standard_error = np.sqrt(p * (1 - p) / N)

print(f"Extinct by day 600: {np.sum(extinct)} out of {N}")
print(f"Estimated extinction probability: {p:.3f} ± {standard_error:.3f} (standard error)")

for margin in [0.01, 0.001]:
    estimated_N = int(np.ceil(p * (1 - p) / margin**2))
    print(
        f"For ±{margin * 100:.1f} percentage point: "
        f"about {estimated_N:,} runs using this estimate of p."
    )

plt.hist(extinction_times[extinct], bins=30)
plt.xlabel("fox extinction time (days)")
plt.ylabel("number of trajectories")
plt.title("Fox extinction times by day 600")
plt.show()

# %% [markdown]
# Extinct by day 600: 2135 out of 3000
# Estimated extinction probability: 0.712 ± 0.008 (standard error)
# For ±1.0 percentage point: about 2,052 runs using this estimate of p.
# For ±0.1 percentage point: about 205,198 runs using this estimate of p.
# Foxes usually die out on average at 275 days 

# %% [markdown]
# ## 6. Your project
#
# One paragraph. Is there any part of your project where the numbers are small
# enough, or the events discrete and random enough, that a continuum model (an ODE or
# a PDE) could mislead you, the way the ODE does about the foxes? Think about
# molecules on a surface, cells, bubbles, particles, defects, failures. If there is,
# what would the "events" and their rates be? If there isn't, why is a continuum model
# good enough?

# %% [markdown]
# If i were say modeling diffusion of molecules through a thin film (what my project could be on), that is very random. A ODE or PDE could be used to describe macrostates and average values but it could be misleading when fluctuations happen. The events could be movement  of molecule on specified locations of the film. Rate could be modeled around the molecules diffusivity (which depends on other variables) and the distance between regions of the film. 

# %% [markdown]
# ## Stretch problems
#
# **(a) Faster.** Time one trajectory. Then find where the time goes (`%prun`, or
# time the pieces). Try Python's `random.random()` in place of `rng.random()`,
# appending to lists versus filling a preallocated array, and recording only every
# 100th event. How much faster can you make it?
#
# **(b) Bigger, on Explorer.** The `.py` paired with this notebook is a script.
# Make a copy that runs 10,000 trajectories and saves the second-peak times and heights
# and the extinction times with `np.savetxt`, and run it as a batch job on Explorer,
# as in Lecture 9. Do your answers to tasks 4 and 5 hold up?
#
# **(c) Is the random clock worth it?** Wikipedia notes that advancing the clock by
# $\Delta t = 1/Q_k$ every step, instead of an exponentially distributed $\Delta t$, gives
# "the same *average* time scale" (and that note is marked *citation needed*). Test it.
# Make a copy of your `kmc` function that uses $\Delta t = 1/Q$, and run both versions
# many times. Compare the probability that the foxes die out by day 600, the fraction of
# runs with a second peak, and the mean time and height of that peak, each with its
# standard error. Which differences are bigger than their uncertainty? How many runs did
# you need before you could tell? (Several thousand of each; (a) and (b) help.) Why might
# some quantities change and others not?

# %% [markdown]
# ---
#
# ### When you're done
#
# Restart & Run All. Check that every cell ran, in order, with no errors. Commit and
# push. Then check on GitHub that your fork shows your latest commit, and that the
# notebook displays with its plots.
