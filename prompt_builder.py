def build_prompt(text):

    style = (
        "police forensic sketch, "
        "criminal suspect portrait, "
        "black and white pencil drawing, "
        "realistic face, "
        "high detail, "
        "front face, "
        "FBI sketch style, "
        "forensic artist drawing, "
    )

    prompt = style + text

    return prompt


if __name__ == "__main__":

    t = input("Enter description: ")

    p = build_prompt(t)

    print("Final prompt:")
    print(p)