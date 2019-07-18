export let lData 
export let cData
export let aData

export function setLData(data){
    lData = (data);
}
export function getLData(){
    return lData;
}
export function setAData(data){
    aData = (data) as Object;
}
export function getAData(){
    return aData;
}
export function setCData(data){
    cData = (data);
}
export function getCData(){
    return cData;
}

// Maps the VideoPLayer's current time frame
// with the corresponding classes of each Model
export function getFrameClasses(index: number, model: string) {
    // Handling undefined index
    if (index >= 0) {
        let data :any
        if(model === 'locale')
            data = lData;
        else if(model === 'cad')
            data = cData;
        else if(model === 'aparallel')
            data = aData
        let parsedData = [];
        if(data[index] && data[index].length){
        for (let i = 0; i < data[index].length; i++) {
            let temp = {
                'class': data[index][i][0],
                'accuracy': data[index][i][1]
            }
            parsedData.push(temp);
        }
    }
        return parsedData
    }
}

//  Generates distinct classes and their 
//  corresponding frequency from the model 

export function  generateClassesFrequency(model) {
    let data :any
        if(model === 'locale')
            data = lData;
        else if(model === 'cad')
            data = cData;
        else if(model === 'aparallel')
            data = aData
    let classesArray = [];
    for (let key in data) {
        for (let i = 0; i < data[key].length; i++) {
            classesArray.push(data[key][i][0])
        }
    }
    const groupByClass = classesArray.reduce((acc, it) => {
        acc[it] = acc[it] + 1 || 1;
        return acc;
    }, {});
    return createDataPoints(groupByClass);
}

// Generates the X,Y mapping of the 
// classes and their frequency from the 
// model required for the CanvasJS library 
// to plot the graph 
export  function createDataPoints(classesFrequency: {}) {
    let mappedPoints = []
    for (let key in classesFrequency) {
        let temp = {
            y: classesFrequency[key],
            label: key
        };
        mappedPoints.push(temp);
    }
   return mappedPoints;
}