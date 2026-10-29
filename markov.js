const DATASET = require("./billboard_chord_sequences");
const POP909_CHORDS = require('./pop909_chord_sequences')
const {transposeSongsToCMajor} = require("./TransposeKey")

class MarkovModel {

    probabilities(obj) {

        const values = Object.values(obj);

        const sum = values.reduce(
            (total, count) => total + count,
            0
        );

        if (sum === 0) {
            return;
        }

        for (const key in obj) {
            obj[key] /= sum;
        }
    }

    find_indexes(item, dataset) {

        const indexes = [];

        dataset.forEach((chord, index) => {

            if (item === chord) {
                indexes.push(index);
            }

        });

        return indexes;
    }

    FirstOrder(data) {

        const obj = {};

        for (const song of data) {

            const chords = song.chords;

    
            for (let i = 0; i < chords.length; i++) {

                const current = chords[i];

                if (!obj[current]) {
                    obj[current] = {};
                }

        

                if (i + 1 < chords.length) {

                    const next = chords[i + 1];

                    if (!obj[current][next]) {
                        obj[current][next] = 1;
                    } else {
                        obj[current][next] += 1;
                    }
                }
            }
        }

        return obj;
    }

    run_first_order(DATA) {

        const obj = this.FirstOrder(DATA);

        for (const key in obj) {

            this.probabilities(obj[key]);
        }

        return obj;
    }
}


const objMM = new MarkovModel();
let  dataset = transposeSongsToCMajor(POP909_CHORDS)

const results = objMM.run_first_order(dataset);
console.log(results)
//first pipeline  - RULE-BASED MODEL, also incorporate key signatures and beats
//TODO second order, third order and multi-variational order, and then create a prediction model using order threshold for chord prediction
//also split the songs in train/test/validation sets for metrics 



