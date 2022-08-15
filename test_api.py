import requests


def test_api(text):
    url = f'http://34.125.241.180:5000/get_text_readability/{text}'
    result = requests.get(url)
    result = result.json()

    print(f'Provided text : {text}')
    print(f'Score: {result["score"]}')


if __name__ == '__main__':
    text = """
    Then the man took off his hat and walked away, and Philip and his sister went home. 
    She seemed different, somehow, and he was sent to bed a little earlier than usual, 
    but he could not go to sleep for a long time, 
    because he heard the front-door bell ring and afterwards a man's voice and Helen's going on 
    and on in the little drawing-room under the room which was his bedroom. He went to sleep at last, 
    and when he woke up in the morning it was raining, and the sky was grey and miserable. 
    He lost his collar-stud, he tore one of his stockings as he pulled it on, 
    he pinched his finger in the door, and he dropped his tooth-mug, with water in it too, 
    and the mug was broken and the water went into his boots. 
    There are mornings, you know, when things happen like that. This was one of them.
    """
    test_api(text)
