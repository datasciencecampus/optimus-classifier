# optimus-classifier



The Optimus classifier is a tool developed by the Data Science Campus to automatically classify lists of items into 42 commodity categories. The classifier predicts categories for items solely from free-text description.

This repository contains a pretrained version of the classifier for public use. 

See a full list of categories at the bottom of this page.

#### Example

#### Setup

There is a `setup.zsh` file in the root directory of this repo. Running this will install the required python libraries, clone and install the Fasttext repo and download the pretrained fasttext model.



#### Running a test

The python code comes with some example dummy data to test that everything is working. The test data is located at `./projects/Optimus-II/data/items.csv`.

1. Open a linux terminal  
2. Type `cd ./projects/Optimus-II/` and hit enter to change directory
3. Type `python3 predict.py ./data/items.csv`

The code should run and return to the linux prompt afterwards. The code produces two new data sets in the `./data/` directory which are the original file name suffixed with `_predictions` and `predictions_pp` respectively.

* ./data/items_predictions.csv
* ./data/items_predictions_pp.csv

The first of these `FILENAME_predictions.csv` is the raw predictions from the nueral network classifier. The second, `FILENAME_predictions_pp.csv` is the raw predictions after being passed through a post-processing step that performs some string matching and replacement to improve upon classifications.


##### Data specification

The data should be prepared in the correct way before being passed to the tool. the requirements are:

1. The csv should only contain the column of item descriptions. This can be copied from a spreadsheet into a new sheet and saved as a csv.
2. The column header should be "original"
3. The items should be converted to uppercase
4. Duplicates should be removed from the data (important for rejoining to original data)
5. Specifying the encoding as `UTF-8` will prevent some encoding errors.


##### Using the tool on new data

To use the tool once the data has been processed as described above, move it to the `./projects/Optimus-II/data/` directory, or you can pass the file location to the tool through the terminal command `python3 predict.py "./SOME/OTHER/FILEPATH/FILENAME.csv"`

Note that the output data will always be placed in the same directory as the source file.

After processing you will need to join the classifications back to the original dataset this can be done using `vlookup` in excel, or a join in `R` or `Python` for example.

#### The classifier


#### Categories

The classifier is traine don the following categories:
* Aerospace
* Animal Feed
* Animal - Agriculture
* Animal - General
* Animal - Pet
* Animal - Zoological
* Beverages
* Building Material
* Ceramics
* Chemicals
* Chemicals - Cosmetics
* Chemicals - Petroleum
* Dairy
* Electrical \ Electronic goods
* Empty
* Fish and crustaceans
* Foods General
* Footwear and Accessories
* Fruit and Vegetables
* Furniture
* General
* Groupage
* Machinery
* Mail
* Meat
* Metal
* Mixed materials
* POAO non-food  (*Product of Animal Origin*)
* Packaging
* Pallets - Empty
* Paper
* Peat
* Pharmaceuticals
* Plants and flowers
* Plastic
* Textile
* Timber
* Toiletries
* Toys
* Unknown
* Vehicle Parts
* Vehicles
* Waste
* Waste - Recycled
