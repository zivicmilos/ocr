# import libraries here
import cv2
import numpy as np
from fuzzywuzzy import fuzz
from keras import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta
    :return: Objekat modela
    """
    letters = []
    for image_path in train_image_paths:
        img_color = load_image(image_path)
        img_gray = image_gray(img_color)
        img = image_bin(img_gray)
        #display_image(img)

        #img = erode(dilate(img))
        #display_image(img)
        selected_regions, l, region_distances = select_roi(img_color.copy(), img)
        letters += l
        #display_image(selected_regions)

    print('Broj prepoznatih regiona:', len(letters))

    inputs = prepare_for_ann(letters)
    outputs = convert_output(get_alphabet())

    # probaj da ucitas prethodno istreniran model
    ann = load_trained_ann()

    # ako je ann=None, znaci da model nije ucitan u prethodnoj metodi i da je potrebno istrenirati novu mrezu
    if ann is None:
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        # serijalizuj novu mrezu nakon treniranja, da se ne trenira ponovo svaki put
        serialize_ann(ann)
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati ako je vec istreniran
    model = ann
    return model


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    img_color = load_image(image_path)
    img_gray = image_gray(img_color)
    img = image_bin_hsv(img_color, None, None)
    #display_image(img)

    #img = erode(dilate(img))
    #display_image(img)
    selected_regions, letters, distances = select_roi(img_color.copy(), img)
    if len(letters) <= 5 or len(letters) > 100:
        img = image_bin_hsv(img_color, (0, 0, 225), (179, 255, 255))
        selected_regions, letters, distances = select_roi(img_color.copy(), img)
    if len(letters) <= 5 or len(letters) > 100:
        img = image_bin_hsv(img_color, (0, 0, 160), (179, 255, 255))
        selected_regions, letters, distances = select_roi(img_color.copy(), img)
    if len(letters) <= 5 or len(letters) > 100:
        img = image_bin_hsv(img_color, (0, 145, 210), (179, 255, 255))
        selected_regions, letters, distances = select_roi(img_color.copy(), img)
    if len(letters) <= 5 or len(letters) > 100:
        img = image_bin_hsv(img_color, (0, 75, 225), (179, 255, 255))
        selected_regions, letters, distances = select_roi(img_color.copy(), img)
    if len(letters) <= 5 or len(letters) > 100:
        img = image_bin_hsv(img_color, (0, 50, 200), (179, 255, 255))
        selected_regions, letters, distances = select_roi(img_color.copy(), img)
    if len(letters) <= 5 or len(letters) > 100:
        img = image_bin_hsv(img_color, (30, 0, 150), (179, 255, 255))
        selected_regions, letters, distances = select_roi(img_color.copy(), img)
    if len(letters) <= 5 or len(letters) > 100:
        img = image_bin(img_gray)
        img = erode(dilate(img))
        selected_regions, letters, distances = select_roi(img_color.copy(), img)
    #display_image(selected_regions)

    """if len(letters) > 200:
        return ''"""

    print('Broj prepoznatih regiona:', len(letters))

    try:
        distances = np.array(distances).reshape(len(distances), 1)

        k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
        k_means.fit(distances)
    except:
        return ''

    inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))

    extracted_text = display_result(results, get_alphabet(), k_means)
    print(extracted_text)
    results = extracted_text.split(' ')

    for i, res in enumerate(results):
        fuzzy_dict = dict()
        for word in vocabulary:
            fuzzy_dict[word] = fuzz.ratio(word, res)
        best_word = max(fuzzy_dict, key=fuzzy_dict.get)
        results[i] = best_word
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string
    extracted_text = ' '.join(results)

    return extracted_text


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 5)
    return image_bin


def image_bin_hsv(image, lower, upper):
    image_conv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = image_conv[:, :, 0], image_conv[:, :, 1], image_conv[:, :, 2]
    #display_image(h)
    #display_image(s)
    #display_image(v)
    reth, image_bin = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)
    #print(reth)
    rets, image_bin = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)
    #print(rets)
    retv, image_bin = cv2.threshold(v, 0, 255, cv2.THRESH_OTSU)
    #print(retv)
    """hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
    plt.plot(hist_h, color='r', label="h")
    plt.legend()
    plt.show()
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")
    """
    if lower is None:
        image_bin = cv2.inRange(image_conv, (reth, rets, 0), (179, 255, 255))
    else:
        image_bin = cv2.inRange(image_conv, lower, upper)

    #image_bin = erode(dilate(image_bin))
    #display_image(image_bin)

    #image_bin = cv2.medianBlur(image_bin, 3)
    return image_bin


def invert(image):
    return 255 - image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def dilate(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


# Funkcionalnost implementirana u OCR basic
def resize_region(region):
    resized = cv2.resize(region, (28, 28), interpolation=cv2.INTER_CUBIC)
    return resized


def scale_to_range(image):
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann


def blocks_of_image(vector):
    new_vector = []
    block_size = 16
    block_width = 4
    width = 32
    for i in range(0, width, block_width):
        for j in range(0, width, block_width):
            s = 0
            for k in range(i, i+block_width):
                for t in range(j, j+block_width):
                    s += vector[k*width+t]
            new_vector.append(s)
    return [el/block_size for el in new_vector]


def hog(image):
    winSize = (28, 28)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (4, 4)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 28
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)
    hist = hog.compute(image, winStride, padding, locations)
    hog_image = hog.compute(image)
    display_image(hist)
    return hog_image


def convert_output(outputs):
    return np.eye(len(outputs))


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def get_alphabet():
    return ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
            'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']


def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        region = image_bin[y:y + h + 1, x:x + w + 1]
        regions_array.append([resize_region(region), (x, y, w, h)])
        # cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 255), 5)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]

    avg_size = sum(rec[2] * rec[3] for rec in sorted_rectangles) / len(sorted_rectangles) / 30 if len(sorted_rectangles) != 0 else 0

    sorted_rectangles = [rec for rec in sorted_rectangles if rec[2] * rec[3] >= avg_size]

    i = 0
    while i < len(sorted_rectangles) - 1:
        if sorted_rectangles[i][0] + sorted_rectangles[i][2] > sorted_rectangles[i + 1][0] and sorted_rectangles[i][1] > \
                sorted_rectangles[i + 1][1] and sorted_rectangles[i][1] - sorted_rectangles[i + 1][1] \
                <= sorted_rectangles[i][3] / 2 \
                and sorted_rectangles[i+1][0] + sorted_rectangles[i+1][2] < sorted_rectangles[i][0] + 1.2*sorted_rectangles[i][2]:
            new_rec = (sorted_rectangles[i][0], sorted_rectangles[i + 1][1], sorted_rectangles[i][2],
                       sorted_rectangles[i][1] + sorted_rectangles[i][3] - sorted_rectangles[i + 1][1])
            sorted_rectangles.pop(i)
            sorted_rectangles.pop(i)
            sorted_rectangles.insert(i, new_rec)
            i -= 1
        i += 1

    sorted_regions = [resize_region(image_bin[rec[1]:rec[1] + rec[3] + 1, rec[0]:rec[0] + rec[2] + 1]) for rec in
                      sorted_rectangles]

    i = 0
    while i < len(sorted_regions) - 1:
        if sorted_rectangles[i][0] <= sorted_rectangles[i + 1][0] and sorted_rectangles[i][1] <= \
                sorted_rectangles[i + 1][1] \
                and sorted_rectangles[i][0] + sorted_rectangles[i][2] >= sorted_rectangles[i + 1][0] + \
                sorted_rectangles[i + 1][2] \
                and sorted_rectangles[i][3] + sorted_rectangles[i][1] >= sorted_rectangles[i + 1][3] + \
                sorted_rectangles[i + 1][1]:
            sorted_rectangles.pop(i + 1)
            sorted_regions.pop(i + 1)
            i -= 1
        i += 1

    for x, y, w, h in sorted_rectangles:
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    #ann.add(Dense(140, activation='sigmoid'))
    ann.add(Dense(80, activation='sigmoid'))
    ann.add(Dense(60, activation='softmax'))
    return ann


def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''

    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.001, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer='adam')

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=500, batch_size=1, verbose=1, shuffle=True)

    return ann


def serialize_ann(ann):
    # serijalizuj arhitekturu neuronske mreze u JSON fajl
    model_json = ann.to_json()
    with open("serialization_folder/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    # serijalizuj tezine u HDF5 fajl
    ann.save_weights("serialization_folder/neuronska.h5")


def load_trained_ann():
    try:
        # Ucitaj JSON i kreiraj arhitekturu neuronske mreze na osnovu njega
        json_file = open('serialization_folder/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        # ucitaj tezine u prethodno kreirani model
        ann.load_weights("serialization_folder/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        # ako ucitavanje nije uspelo, verovatno model prethodno nije serijalizovan pa nema odakle da bude ucitan
        return None


def display_result(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]

    return result


if __name__ == '__main__':
    model = train_or_load_character_recognition_model(['dataset/train/alphabet0.bmp', 'dataset/train/alphabet1.bmp'])
    vocabulary = dict()
    with open('dataset/dict.txt', 'r', encoding='utf-8') as file:
        data = file.read()
        lines = data.split('\n')
        for index, line in enumerate(lines):
            cols = line.split()
            if len(cols) == 3:
                vocabulary[cols[1]] = cols[2]
    s = extract_text_from_image(model, 'dataset/validation/train63.bmp', vocabulary)
    print(s)
    #image_bin_hsv(load_image('dataset/validation/train88.bmp'))
