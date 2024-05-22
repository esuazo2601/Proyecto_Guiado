if __name__ == '__main__':
    
    from NEAT.neat import NEAT
    model = NEAT(
        inputSize=128,
        outputSize=6,
        populationSize=120,
        C1=1.0,
        C2=2.0,
        C3=3.0
    )

    # Define la función de entrenamiento y entrena el modelo
    model.train(
        epochs=400,
        goal=800,
        distance_t=0.3,
        output_file="fitness_history.txt"
    )
