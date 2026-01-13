from inspect import getsource
from shiny import reactive

print('members:', [m for m in dir(reactive.Value) if not m.startswith('_')])
print('\nSource of reactive.Value.get:\n')
print(getsource(reactive.Value.get))
print('\nSource of reactive.Value.__call__:\n')
print(getsource(reactive.Value.__call__))
