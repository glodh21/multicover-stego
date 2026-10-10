from image_percent import ImageCapacityCalculator


def main():
    # Put your image paths here
    my_images = [
        "images/high_bike.png",
        "images/low-sky.png",
        "images/mid_house.png",
        "images/mid_door.png",
        "images/high_kayak.png",
    ]

    calculator = ImageCapacityCalculator(floor_fraction=0.03)
    capacities = calculator.calculate_image_capacities(my_images)
    calculator.print_results(capacities)


if __name__ == "__main__":
    main()