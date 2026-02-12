# Homework Extension Project

There are 10 points possible for this assignment. Please carefully read this
entire README.

## Academic integrity

Your work must be your own. You may not work with others. Do not submit other
people's work as your own, and do not allow others to submit your work as
theirs. 

If you need help debugging your code, make a *private* post on Piazza or come
to office hours. You may not show your code (including pseudocode) to other
students under any circumstances.

You are required to completely understand any homework solution that you
submit, and, in case of any doubt, you must be prepared to orally explain your
solution. If you have submitted a solution that you cannot verbally explain,
then you have violated this policy.

Please see [the academic integrity
policy](https://canvas.northwestern.edu/courses/246595/pages/academic-integrity)
for more detail. By submitting your work to Canvas or pushing your code to
GitHub, you agree to these rules, and understand that there may be severe
consequences for violating them.

## Goals and structure

The purpose of this assignment is to give you an opportunity for an open-ended
exploration of topics in machine learning that goes beyond what the assignments
can cover. Given the scale of this course we cannot provide extensive
one-on-one feedback throughout that exploration, but the assignment will be
structured into two submissions:

- your proposal, due on February 25
- your final report, due March 12

## Late work

You may use your late days for this assignment. However, the ability to revise
your final report is only possible if you submit it before March 10, and that
cannot be extended by late days.

## Proposal

Your proposal should be a brief description of what you want to explore and
what work you plan to accomplish. It should be no longer than one page.  Your
proposal itself will not be graded, but we will give you feedback on the scope
and feasibility of what you plan to do. In particular, if your proposed work
seems too difficult, we will try to warn you of possible pitfalls. If your work
seems insufficiently substantive, we will estimate of the maximum number of
points we expect your proposed work would be able to earn.

If you want to change your project based on our feedback, you do not need to
tell us until you submit your final report. However, you're welcome to
communicate with the course staff via Piazza or email to ask for help or
clarification. If you submit your proposal early, we will have more time to
provide feedback and guidance.

If you do not submit a proposal, you may not earn points for the final report.

## Final report

Your final report should be a concise description of all the work you've
completed as part of this project. It should be no longer than two pages, not
including references. You should push any code you write to this GitHub
repository, and you are welcome to include any appendices or additional written
content here. However, you should expect us to base your grade solely on the
material described and presented in your two-page final report.

If you submit your final report on or after March 12, you will not have the
chance to revise it. If you submit your report no later than March 10, we will
grade your work within one week and give you at least 48 hours to submit a
revision based on our feedback.

## Project ideas

This project is meant to be open-ended, so you are encouraged to come up with
your own ideas based on what is most interesting to you. You are welcome to use
the following ideas as inspiration, but your proposal should provide much more
detail than the bullet points below.

### Extending Homework 1

- Implement at least two ways to constrain your DecisionTree's complexity
  (e.g., `max_depth`, `min_impurity_decrease`) and evaluate how they control
  the trade-off between overfitting and underfitting on one real-world dataset.
- Implement a RandomForestClassifier that uses your DecisionTree model as its
  base estimator, and compare it against `scikit-learn`'s implementation.
- Change the ID3 algorithm to incorporate a fairness constraint (see HW1 FRQ4)
  or to allow it to use the same feature more than once. Compare this
  implementation against your original HW1 implementation on at least one
  real-world dataset.

### Extending Homework 2

- Implement L1 and L2 regularization for your regression model(s) and evaluate
  how they control overfitting as you increase the degree of your polynomial
  feature transformation.

### Extending Homework 3

- Implement the `forward` and `backward` functions for convolutional layer 
  and use it to build a simple CNN.
- Implement Dropout or Batch Normalization as additional regularization methods
  and evaluate how they control the trade-off between overfitting and
  underfitting on one real-world dataset.

### Extending Homework 4

- Extend your NaiveBayes implementation by adding regularization and a
  mechanism for calibrating probabilities on a validation dataset.

## Ask for help

This is an open-ended project and Winter 2026 is the first time we are
including this as a component of the course. If you have questions or feedback
about the assignment, please share it with us via Piazza, at office hours,
or in anonymous surveys.
