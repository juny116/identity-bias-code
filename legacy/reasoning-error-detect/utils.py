def messages_to_string(messages):
    text = ""
    for message in messages:
        text += message["content"]
    return text


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num
