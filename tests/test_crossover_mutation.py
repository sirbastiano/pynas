from crossover import single_point_crossover
from mutation import gene_mutation




parent1 = pop[0]
parent2 = pop[1]

for item1, item2 in zip(parent1.parsed_layers,parent2.parsed_layers):
    print(item1['layer_type'], "||", item2['layer_type'])

children = single_point_crossover([parent1, parent2])

print('\n\nNew Generation:')

for item1, item2 in zip(children[0].parsed_layers, children[1].parsed_layers):
    print(item1['layer_type'], "||",item2['layer_type'])
    
    
mutated_child = gene_mutation(children[0], 0.5)
print("Before mutation:", children[0].parsed_layers)
print("After mutation:", mutated_child.parsed_layers)


print("Before mutation:", children[0].chromosome)
print("After mutation:", mutated_child.chromosome)