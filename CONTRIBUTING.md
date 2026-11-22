# Contributing: how to add a new learning

This guide stays useful by growing. A new lesson is welcome when it saved or cost you a run. Every lesson has four parts. A
lesson without all four is a rumour, and it does not go in.

1. **Symptom.** What you saw, in the words you would have used before you knew the cause. ("The model answered with `!`.")
2. **Cause.** What was really wrong, and how you know (what you changed, what you compared).
3. **Fix.** The smallest change that removed the symptom, as a command or a setting a reader can copy.
4. **Where it was measured.** The number, what it measures, and on what: the machine, the tool version, the size of the test
   set, how many runs. "Pass rate fell from 20.6% to 9.4%" is not enough on its own; say what the pass rate is a rate of.

## Rules

- **Only numbers you measured.** If you are unsure of a number, leave it out and say what you did not measure.
- **Say what a word means.** "Step", "epoch", "iteration" and "update" differ between tools; write lengths as rows seen and passes.
- **Two numbers.** If a result is a score, give the strict one and the usable one, and say which test file it is.
- **Keep it generic.** No private paths, machine names, hostnames, addresses, keys, ticket numbers, or personal names. Describe
  machines by what limits them ("an 8 GB MacBook", "a GPU with 4 GB").
- **Do not describe how a private dataset was made.** The guide teaches the method; it is not a project diary.
- **Plain, short sentences.** First person is fine where it is your experience.

## Where it goes

- A mistake goes into chapter 8 of `GUIDE.md` (the next number), with the four parts as bullets; add one line to the checklist in
  chapter 9 and, if it has a number, one row to the appendix table.
- A setting or version that changes goes where it is first used, and into `kit/requirements-*.txt` if it is a version.
- A script that you needed goes into `kit/`: standard library where you can, a docstring that says how to run it, and a line in
  `kit/README.md`. It must pass `kit/smoke.sh --dry-run`.
- Add an entry to `CHANGELOG.md` (bump the minor version for a new lesson, the patch version for a fix).

## Before you open a pull request

```sh
kit/smoke.sh --dry-run
grep -rniE "/Users/|/home/[a-z]|password|token|api[_-]?key" . --include='*.md' --include='*.py' --include='*.sh'
```

The second command should print nothing you would not want a stranger to read.
