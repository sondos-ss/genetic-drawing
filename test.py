from PIL import Image
from evol import Evolution, Population
from painting import Painting
import os
import random
from copy import deepcopy

# --- الصورة الهدف أصغر للتجربة ---
original_image_path = "./img/Starry_Night.jpg"
img = Image.open(original_image_path)

# صغر الصورة لنصف الحجم
width, height = img.size
img_small = img.resize((width // 2, height // 2))

# احفظي الصورة المصغرة باسم جديد
target_image_path = "./img/starry_night_half.jpg"
img_small.save(target_image_path)
print("تم إنشاء الصورة المصغرة!")

# مجلد لحفظ الصور المؤقتة والتطور
checkpoint_path = "./fast_run/"
os.makedirs(checkpoint_path, exist_ok=True)

# --- دوال الدعم ---
def score(painting: Painting) -> float:
    if not painting.target_image_path:
        painting.target_image_path = target_image_path
    target = Image.open(painting.target_image_path).convert("RGBA")
    return painting.image_diff(target)

def pick_best_and_random(pop, maximize=False):
    evaluated_individuals = tuple(filter(lambda x: x.fitness is not None, pop))
    if evaluated_individuals:
        mom = max(evaluated_individuals, key=lambda x: x.fitness if maximize else -x.fitness)
    else:
        mom = random.choice(pop)
    dad = random.choice(pop)
    return mom, dad

def mutate_painting(p: Painting, rate=0.04, swap=0.5, sigma=1) -> Painting:
    p.mutate_triangles(rate=rate, swap=swap, sigma=sigma)
    return deepcopy(p)

def mate(mom: Painting, dad: Painting):
    child_a, _ = Painting.mate(mom, dad)
    return deepcopy(child_a)

def print_summary(pop, img_template="output_%03d.png", checkpoint_path="output") -> Population:
    avg_fitness = sum(i.fitness for i in pop.individuals) / len(pop.individuals)
    print(f"\nGeneration {pop.generation}, best score {pop.current_best.fitness:.2f}, avg {avg_fitness:.2f}")
    img = pop.current_best.chromosome.draw()
    img.save(os.path.join(checkpoint_path, img_template % pop.generation), 'PNG')

    if pop.generation % 5 == 0:  # كل 5 أجيال تحفظ الصورة
        pop.checkpoint(target=checkpoint_path, method='pickle')

    return pop

# --- إعداد السكان والتطور ---
num_triangles = 50
population_size = 50

pop = Population(
    chromosomes=[Painting(num_triangles, target_image_path, background_color=(255, 255, 255)) 
                 for _ in range(population_size)],
    eval_function=score,
    maximize=False,
    concurrent_workers=4
)

evolution = (Evolution()
             .survive(fraction=0.1)
             .breed(parent_picker=pick_best_and_random, combiner=mate, population_size=population_size)
             .mutate(mutate_function=mutate_painting, rate=0.05, swap=0.25)
             .evaluate(lazy=False)
             .callback(print_summary, img_template="fast_%03d.png", checkpoint_path=checkpoint_path)
)

# تشغيل التطور لعدد أقل من الأجيال للتجربة
pop = pop.evolve(evolution, n=200)
