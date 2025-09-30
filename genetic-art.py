import os
import random
from copy import deepcopy
from PIL import Image
from evol import Evolution, Population
from painting import Painting


# =======================
# Evaluation function for Painting
# =======================
def score(painting: Painting) -> float:
    """
    Calculate the difference between the current image and the target image.
    """
    if painting.target_image_path is None:
        raise ValueError("Painting has no target image path")
    
    target_image = Image.open(painting.target_image_path).convert("RGBA")
    current_score = painting.image_diff(target_image)
    print(".", end='', flush=True)  # print progress indicator
    return current_score


# =======================
# Parent selection for mating
# =======================
def pick_best_and_random(pop, maximize=False):
    """
    Pick the best individual (mom) and a random individual (dad) from the population.
    """
    evaluated_individuals = tuple(filter(lambda x: x.fitness is not None, pop))
    if len(evaluated_individuals) > 0:
        mom = max(evaluated_individuals, key=lambda x: x.fitness if maximize else -x.fitness)
    else:
        mom = random.choice(pop)
    dad = random.choice(pop)
    return mom, dad


# =======================
# Mutation operation
# =======================
def mutate_painting(painting: Painting, rate=0.04, swap=0.5, sigma=1.0) -> Painting:
    """
    Mutate a Painting by altering some of its triangles.
    """
    painting.mutate_triangles(rate=rate, swap=swap, sigma=sigma)
    return deepcopy(painting)


# =======================
# Crossover operation
# =======================
def mate(mom: Painting, dad: Painting) -> Painting:
    """
    Create a child Painting from parent A (mom) and parent B (dad).
    """
    child_a, _ = Painting.mate(mom, dad)
    return deepcopy(child_a)


# =======================
# Logging and saving progress
# =======================
def print_summary(pop, img_template="output%d.png", checkpoint_path="output") -> Population:
    """
    Print information about the current generation and save the best image.
    """
    avg_fitness = sum([i.fitness for i in pop.individuals]) / len(pop.individuals)
    print("\nCurrent generation %d, best score %f, pop. avg. %f " %
          (pop.generation, pop.current_best.fitness, avg_fitness))

    img = pop.current_best.chromosome.draw()
    os.makedirs(checkpoint_path, exist_ok=True)
    img.save(os.path.join(checkpoint_path, img_template % pop.generation), 'PNG')

    # Save population every 50 generations
    if pop.generation % 50 == 0:
        pop.checkpoint(target=checkpoint_path, method='pickle')

    return pop


# =======================
# Main entry point
# =======================
if __name__ == "__main__":
    # Paths
    target_image_path = "./img/starry_night_half.jpg"
    checkpoint_path = "./out/"
    image_template = "drawing_%05d.png"

    # Open target image
    target_image = Image.open(target_image_path).convert("RGBA")

    # Initialize population
    num_triangles = 50
    population_size = 50
    pop = Population(
        chromosomes=[Painting(num_triangles, target_image_path, background_color=(255, 255, 255)) 
                     for _ in range(population_size)],
        eval_function=score,
        maximize=False,
        concurrent_workers=6
    )

    # Define evolution steps
    evolution = (Evolution()
                 .survive(fraction=0.05)
                 .breed(parent_picker=pick_best_and_random, combiner=mate, population_size=population_size)
                 .mutate(mutate_function=mutate_painting, rate=0.05, swap=0.25)
                 .evaluate(lazy=False)
                 .callback(print_summary, img_template=image_template, checkpoint_path=checkpoint_path))

    # Start evolution
    pop = pop.evolve(evolution, n=5000)
