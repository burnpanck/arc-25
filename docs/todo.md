# TODO

## vision
- implement attention dropout `FieldAttention`
- try symmetry-preserving dropout?
- implement stochastic layer drop
- train semantic grid contents directly clip style

## reasoning agent
- Implement solution code outputing two alternate solutions
  - re-write some training example solutions to make use of that (horizontal/vertical are go-to's)
- Implement managing multiple alternate solutions in the DB and the explorer
- Implement managing iterative improval of a solution in the DB and the explorer
- Speculative decoding at inference!

# Pending refactors
- Get rid of `etils.epath` in favour of `upath` (etils is way too eager to rely on TensorFlow)
- Migrate away from `pmap`, probably just to automatic parallelisation
