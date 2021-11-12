//const threshold = 0.3;
const person_class = 1;
var distance_treshold = 0.20;
var image_name = "No file chosen";
var imageLoaded = false;
var ssd_inferred_time = 0;
var centernet_inferred_time = 0;
var message_nonviolation = "No person is detected to violate the social distance";
var message_violation = "are detected to violate the social distance and highlighted in red"

$("#image-selector").change(function () {
	imageLoaded = false;
	clearVariablesForImageButton();
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selectedImage").attr("src", dataURL);
		
		imageLoaded = true;
	}

	let file = $("#image-selector").prop('files')[0];
	console.log(file.name);
	image_name = file.name;
	reader.readAsDataURL(file);

});
$('#distance-threshold-selector').change(function () {
	distance_treshold = $("#distance-threshold-selector option:selected").text();
	console.log(distance_treshold);
});
$(document).ready(async function () {

	$('#ssd_progress_bar').hide();
	$('#centernet_progress_bar').hide();
	$('#ssd_inferrence_content').hide();
	$('#centernet_inferrence_content').hide();
	$('#centroid-chart').hide();
	$('#time-chart').hide();


});

function showSSDProgress(percentage) {
	var pct = Math.floor(percentage * 100.0);
	$('#ssd_progress_bar').html(`Loading SSD Model (${pct}%)`);
	console.log(`${pct}% loaded`);
}

function showEfficientDetProgress(percentage) {
	var pct = Math.floor(percentage * 100.0);
	$('#centernet_progress_bar').html(`Loading EfficientDet Model (${pct}%)`);
	console.log(`${pct}% loaded`);
}


async function loadModel(modelname) {
	let model;
	if (modelname == "ssd") {
		console.log("Loading SSD model...");
		$('#ssd_progress_bar').html("Loading MobileNet Model");
		$('#ssd_progress_bar').show();
		model = await tf.loadGraphModel('model/ssd/model.json', { onProgress: showSSDProgress });
		console.log("Model SSD loaded.");

	}
	else {
		console.log("Loading CenterNet model...");
		$('#centernet_progress_bar').html("Loading EfficientDet Model");
		$('#centernet_progress_bar').show();
		//model = await tf.loadGraphModel('model/centernet/model.json', { onProgress: showEfficientDetProgress });
		model = await tf.loadGraphModel('model/centernet/model.json', { onProgress: showEfficientDetProgress });
		console.log("Model CenterNet loaded.");
	}
	return model;

}


async function loadImage(modelname,model) {
	console.log("Pre-processing image...");
	if (modelname == "ssd") {
		await $('#ssd_progress_bar').html("Pre-processing image").promise();
	}
	else {
		await $('#centernet_progress_bar').html("Pre-processing image").promise();
	}

	const pixels = $('#selectedImage').get(0);
	let image;
	//image = tf.browser.fromPixels(pixels,3).toInt();

	if (modelname == "ssd") {
		image = tf.browser.fromPixels(pixels).toInt();	
		image=image.expandDims();	

	}
	else {
		var inputsize=model.inputs[0].shape[1];
		image = tf.browser.fromPixels(pixels).toFloat();		
		image = tf.image.resizeBilinear(image.expandDims().toFloat(), [inputsize,inputsize]);
	}

	
	
	


	return image;
}
var centroids_non_normalized_ssd = [];
var centroids_non_normalized_centernet = [];
function FilterDetectedObjects(scores, boxes, classes, filter_threshold, filter_class, modelname) {
	const filtered_detectons = [];
	scores[0].forEach((score, i) => {
		if (score > filter_threshold && classes[0][i] == filter_class) {
			const bbox = [];
			bbox[0] = boxes[0][i][0]; //min-y
			bbox[1] = boxes[0][i][1]; //min-x
			bbox[2] = boxes[0][i][2]; //max-y
			bbox[3] = boxes[0][i][3]; //max-x
			if (modelname == "ssd") {
				let centroid = calculate_centroid(bbox);
				centroids_non_normalized_ssd.push(centroid);
			}
			else {
				let centroid = calculate_centroid(bbox);
				centroids_non_normalized_centernet.push(centroid);

			}
			filtered_detectons.push({
				class: classes[0][i],
				score: score,
				bbox: bbox

			});


		}
	})
	return filtered_detectons;
}


async function getPrediction(inputs, model, modelname) {
	console.log("Running distance prediction...");
	if (modelname == "ssd") {
		await $('#ssd_progress_bar').html("Running distance prediction").promise();
	}
	else {
		await $('#centernet_progress_bar').html("Running distance prediction").promise();

	}
	const outputs = await model.executeAsync(inputs);

	const arrays = !Array.isArray(outputs) ? outputs.array() : Promise.all(outputs.map(t => t.array()));
	let predictions = await arrays;


	console.log("model output detail:" + predictions.length);
	if (modelname == "ssd") {
	console.log("predictions[0]:" + predictions[0]);
	console.log("predictions[1]:" + predictions[1]);
	console.log("predictions[2]:" + predictions[2]);
	console.log("predictions[3]:" + predictions[3]);
	console.log("predictions[4]:" + predictions[4]);
	console.log("predictions[5]:" + predictions[5]);
	console.log("predictions[6]:" + predictions[6]);
	console.log("predictions[7]:" + predictions[7]);

	}
	else{
	console.log("predictions[0]:" + predictions[0]);
	console.log("predictions[1]:" + predictions[1]);
	console.log("predictions[2]:" + predictions[2]);
	console.log("predictions[3]:" + predictions[3]);
	}
	return predictions;
}


var children = [];
function removeHighlights() {
	for (let i = 0; i < children.length; i++) {
		imageOverlay.removeChild(children[i]);
	}
	children = [];
}
function clearVariablesForImageButton() {
	removeHighlights();
	centroids_non_normalized_ssd = [];
	centroids_non_normalized_centernet = [];
	ssd_inferred_time = 0;
	centernet_inferred_time = 0;
	$('#ssd_inferrence_content').hide();
	$('#centernet_inferrence_content').hide();

	$('#centroid-chart').hide();
	$('#time-chart').hide();

}
function clearVariablesForEfficientDetPredictionButton() {
	removeHighlights();
	centroids_non_normalized_centernet = [];
	$('#centernet_inferrence_content').hide();
	//Clear the inference values for centernet
	$("#centernet_table").empty();
	$("#centernet_image_name").empty();
	$("#centernet_infer_time").empty()
	$("#centernet_distance_threshold").empty();
	$("#centernet_number_person").empty();
	$("#centernet_centroids").empty();
	$("#centernet_infer_result").empty();

	$('#centroid-chart').hide();
	$('#time-chart').hide();
}
function clearVariablesForMobileNetPredictionButton() {
	removeHighlights();
	centroids_non_normalized_ssd = [];
	$('#ssd_inferrence_content').hide();


	//Clear the inference values for ssd
	$("#ssd_table").empty();
	$("#ssd_image_name").empty();
	$("#ssd_infer_time").empty()
	$("#ssd_distance_threshold").empty();
	$("#ssd_number_person").empty();
	$("#ssd_centroids").empty();
	$("#ssd_infer_result").empty();


	$('#centroid-chart').hide();
	$('#time-chart').hide();


}

async function getInferenceAndhighlightResult(predictions, modelname) {
	const violated_index_list = [];
	var filtered_detectons=[];
	console.log("Inferring and highlighting results...");
	if (modelname == "ssd") {
		await $('#ssd_progress_bar').html("Inferring and highlighting results").promise();
		const scores = predictions[5]
		const classes = predictions[6];
		const boxes = predictions[4];
		 filtered_detectons = this.FilterDetectedObjects(scores, boxes, classes, 0.60, 1, modelname);
	}
	else {
		await $('centernet_progress_bar').html("Inferring and highlighting results").promise();
		const scores = predictions[0]
		const classes = predictions[2];
		const boxes = predictions[3];
		filtered_detectons = this.FilterDetectedObjects(scores, boxes, classes, 0.30, 0, modelname);
	}





	let used_list = [];
	let dist;
	let rowIdx = 0;


	for (let i = 0; i < filtered_detectons.length; i++) {
		used_list.push(i);

		for (let j = 0; j < filtered_detectons.length; j++) {
			if (i != j && !used_list.includes(j)) {

				if (modelname == "ssd") {
					//centroids_non_normalized_ssd=centroids_non_normalized_ssd.sort();
					dist = calculate_centr_distances(centroids_non_normalized_ssd[i], centroids_non_normalized_ssd[j]);
					$('#ssd_table').append('<tr><td>' + (++rowIdx) + '</td><td><p>(' + (i + 1) + ',' + (j + 1)
						+ ')</p></td><td><p>(' + parseFloat(centroids_non_normalized_ssd[i][0]).toFixed(3) + ',' + parseFloat(centroids_non_normalized_ssd[i][1]).toFixed(3) + ')'
						+ '</p></td><td><p>(' + parseFloat(centroids_non_normalized_ssd[j][0]).toFixed(3) + ',' + parseFloat(centroids_non_normalized_ssd[j][1]).toFixed(3) + ')'
						+ '</p></td><td>' + parseFloat(dist).toFixed(3) + '</td></tr>');
				}
				else {
					//centroids_non_normalized_centernet=centroids_non_normalized_centernet.sort();
					dist = calculate_centr_distances(centroids_non_normalized_centernet[i], centroids_non_normalized_centernet[j]);
					$('#centernet_table').append('<tr><td>' + (++rowIdx) + '</td><td><p>(' + (i + 1) + ',' + (j + 1)
						+ ')</p></td><td><p>(' + parseFloat(centroids_non_normalized_centernet[i][0]).toFixed(3) + ',' + parseFloat(centroids_non_normalized_centernet[i][1]).toFixed(3) + ')'
						+ '</p></td><td><p>(' + parseFloat(centroids_non_normalized_centernet[j][0]).toFixed(3) + ',' + parseFloat(centroids_non_normalized_centernet[j][1]).toFixed(3) + ')'
						+ '</p></td><td>' + parseFloat(dist).toFixed(3) + '</td></tr>');

				}
				if (dist < distance_treshold) {
					//console.log("centroids and distance:"+centroids_non_normalized[i]+" "+centroids_non_normalized[j]+" "+dist);
					//console.log("i, j is"+i+" "+j);
					if (!violated_index_list.includes(i)) {
						violated_index_list.push(i);
					}
					if (!violated_index_list.includes(j)) {
						violated_index_list.push(j);
					}
				}
			}
		}

	}
	console.log("the number of distance violation is:" + violated_index_list.length);
	console.log("violated_index_list:" + violated_index_list);
	//const final_normal_index_list = temp_normal_index_list.filter(i => !violated_index_list.includes(i));
	//console.log("final_normal_index_list:" + final_normal_index_list);
	//Draw boxes for normal social distance
	filtered_detectons.forEach((item, i) => {
		if (!violated_index_list.includes(i)) {
			const p = document.createElement('p');
			p.setAttribute('class', 'headerNormalP');
			p.innerText = TARGET_CLASSES[item["class"]] + ': '
				+ Math.round(parseFloat(item["score"]) * 100)
				+ '%';

			bboxTop = (item['bbox'][0] * selectedImage.height) - 10;
			bboxLeft = (item['bbox'][1] * selectedImage.width) + 10;
			bboxHeight = (item['bbox'][2] * selectedImage.height) - bboxTop + 10;
			bboxWidth = (item['bbox'][3] * selectedImage.width) - bboxLeft + 20;


			p.style = 'margin-left: ' + bboxLeft + 'px; margin-top: '
				+ (bboxTop - 10) + 'px; width: '
				+ bboxWidth + 'px; top: 0; left: 0;';
			const highlighter = document.createElement('div');
			highlighter.setAttribute('class', 'highlighter');
			highlighter.style = 'left: ' + bboxLeft + 'px; top: '
				+ bboxTop + 'px; width: '
				+ bboxWidth + 'px; height: '
				+ bboxHeight + 'px;';
			imageOverlay.appendChild(highlighter);
			imageOverlay.appendChild(p);
			children.push(highlighter);
			children.push(p);
		}



	});
	//Draw boxes for violated social distance
	filtered_detectons.forEach((item, i) => {
		if (violated_index_list.includes(i)) {
			const p = document.createElement('p');
			p.setAttribute('class', 'headerViolateP');
			p.innerText = TARGET_CLASSES[item["class"]] + ': '
				+ Math.round(parseFloat(item["score"]) * 100)
				+ '%';

			bboxTop = (item['bbox'][0] * selectedImage.height) - 10;
			bboxLeft = (item['bbox'][1] * selectedImage.width) + 10;
			bboxHeight = (item['bbox'][2] * selectedImage.height) - bboxTop + 10;
			bboxWidth = (item['bbox'][3] * selectedImage.width) - bboxLeft + 20;


			p.style = 'margin-left: ' + bboxLeft + 'px; margin-top: '
				+ (bboxTop - 10) + 'px; width: '
				+ bboxWidth + 'px; top: 0; left: 0;';
			const highlighter = document.createElement('div');
			highlighter.setAttribute('class', 'violatehighlighter');
			highlighter.style = 'left: ' + bboxLeft + 'px; top: '
				+ bboxTop + 'px; width: '
				+ bboxWidth + 'px; height: '
				+ bboxHeight + 'px;';
			imageOverlay.appendChild(highlighter);
			imageOverlay.appendChild(p);
			children.push(highlighter);
			children.push(p);
		}


	});


	return violated_index_list;


}

$("#ssd_predict-button").click(async function () {
	clearVariablesForMobileNetPredictionButton();
	let modelname = "ssd";
	if (!imageLoaded) { alert("Please select an image first"); return; }
	$('#ssd_progress_bar').html("Starting distance prediction");
	$('#ssd_progress_bar').show();
	let startTime = new Date();
	const model = await loadModel(modelname);
	console.log("SSD:" + model.inputs);
	console.log("SSD input 1:" + model.inputs[0].shape);
	console.log("SSD input 2:" + model.inputs[1]);
	const image = await loadImage(modelname,model);
	const predictions = await getPrediction(image, model, modelname);
	const listOfViolatedPerson = await getInferenceAndhighlightResult(predictions, modelname);
	let endTime = new Date();
	ssd_inferred_time = getElapsedTime(startTime, endTime);
	getInferenceInfo(listOfViolatedPerson, modelname);
	$('#ssd_progress_bar').hide();
	$('#ssd_inferrence_content').show();
	console.log("ssd_inferred_time is:" + ssd_inferred_time);


});
$("#centernet_predict-button").click(async function () {
	clearVariablesForEfficientDetPredictionButton();
	//let modelname = "centernet";
	let modelname = "centernet";
	if (!imageLoaded) { alert("Please select an image first"); return; }
	$('#centernet_progress_bar').html("Starting distance prediction");
	$('#centernet_progress_bar').show();
	let startTime = new Date();
	const model = await loadModel(modelname);
	console.log("CenterNet:" + model.inputs);
	console.log("CenterNet input 1:" + model.inputs[0].shape);
	console.log("CenterNet input 2:" + model.inputs[1]);
	const image = await loadImage(modelname,model);
	const predictions = await getPrediction(image, model, modelname);
	const listOfViolatedPerson = await getInferenceAndhighlightResult(predictions, modelname);
	let endTime = new Date();
	centernet_inferred_time = getElapsedTime(startTime, endTime);
	getInferenceInfo(listOfViolatedPerson, modelname);
	$('#centernet_progress_bar').hide();
	$('#centernet_inferrence_content').show();



});
$("#btn_chart_compare").click(async function () {
	if (!imageLoaded) { alert("Please select an image first"); return; }
	if (ssd_inferred_time == 0 || centernet_inferred_time == 0) { alert("Please complete prediction using two models before comparison"); return; }
	console.log(ssd_inferred_time + " " + centernet_inferred_time);
	$('#time-chart').show();
	$('#centroid-chart').show();
	draw_time_chart();
	draw_centroid_chart();

});
function getInferenceInfo(listOfViolatedPerson, modelname) {
	if (modelname == "ssd") {

		$("#ssd_image_name").text(image_name);
		$("#ssd_infer_time").text(ssd_inferred_time + " second(s)");
		$("#ssd_distance_threshold").text(distance_treshold);
		$("#ssd_number_person").text(centroids_non_normalized_ssd.length);
		centroids_non_normalized_ssd.forEach(centroid => { $("#ssd_centroids").append("(" + parseFloat(centroid[0]).toFixed(3) + "," + parseFloat(centroid[1]).toFixed(3) + ") "); });
		if (listOfViolatedPerson.length == 0) {

			$("#ssd_infer_result").text(message_nonviolation);
		}
		else {

			listOfViolatedPerson = listOfViolatedPerson.map(i => i + 1);
			$("#ssd_infer_result").text(listOfViolatedPerson.length + " person(s) with the index (" + listOfViolatedPerson + ") " + message_violation);

		}

	}
	else {
		
		$("#centernet_image_name").text(image_name);
		$("#centernet_infer_time").text(centernet_inferred_time + " second(s)");
		$("#centernet_distance_threshold").text(distance_treshold);
		$("#centernet_number_person").text(centroids_non_normalized_centernet.length);
		centroids_non_normalized_centernet.forEach(centroid => { $("#centernet_centroids").append("(" + parseFloat(centroid[0]).toFixed(3) + "," + parseFloat(centroid[1]).toFixed(3) + ") "); });

		if (listOfViolatedPerson.length == 0) {

			$("#centernet_infer_result").text(message_nonviolation);
		}
		else {

			listOfViolatedPerson = listOfViolatedPerson.map(i => i + 1);
			$("#centernet_infer_result").text(listOfViolatedPerson.length + " person(s) with the index (" + listOfViolatedPerson + ") " + message_violation);

		}

	}
}
function getElapsedTime(startTime, endTime) {
	console.log("startTime is:" + startTime);
	console.log("endTime is:" + endTime);
	let timeDiff = endTime - startTime; //in ms
	timeDiff /= 1000;
	let elapsedTime = Math.round(timeDiff);
	return elapsedTime;
}

function draw_centroid_chart() {

	const ssd_centroids_list = [];
	const centernet_centroids_list = [];
	centroids_non_normalized_ssd.forEach(centroid => { ssd_centroids_list.push({ x: parseFloat(centroid[0]).toFixed(3), y: parseFloat(centroid[1]).toFixed(3) }); });
	centroids_non_normalized_centernet.forEach(centroid => { centernet_centroids_list.push({ x: parseFloat(centroid[0]).toFixed(3), y: parseFloat(centroid[1]).toFixed(3) }); });
	console.log(ssd_centroids_list);

	values = [
		ssd_centroids_list,
		centernet_centroids_list
	];
	series = ['SSD MobileNet v2', 'CenterNet MobileNet V2'];
	tfvis.render.scatterplot(
		document.getElementById('centroid-chart'),
		{ values, series },
		{

			xLabel: 'Centroid x-values',
			yLabel: 'Centroid y-values'
		});

}

function draw_time_chart() {
	const data = [
		{ index: 'SSD MobileNet v2', value: ssd_inferred_time }, { index: 'CenterNet MobileNet V2', value: centernet_inferred_time }];
	tfvis.render.barchart(document.getElementById('time-chart'), data, {
		yLabel: 'Elapsed Inferred Time'

	});

}

function calculate_perm(centroids) {
	let permutations = [];

	let n = centroids.length;
	let used_list = [];

	for (let i = 0; i < n; i++) {


		for (let j = 0; j < n; j++) {
			if (i != j && !used_list.includes(j)) {
				permutations.push([centroids[i], centroids[j]]);
			}
		}
		used_list.push(i);
	}

	return permutations;
}

function calculate_centroid(bounding_box) {
	let centroid = [];
	centroid.push(((bounding_box[3] - bounding_box[1]) / 2) + bounding_box[1], ((bounding_box[2] - bounding_box[0]) / 2) + bounding_box[0]);
	console.log("centroid in calculate_centroid is:" + centroid);
	return centroid;
}

function calculate_centr_distances(centroid_1, centroid_2) {
	let distance = Math.sqrt((centroid_2[0] - centroid_1[0]) ** 2 + (centroid_2[1] - centroid_1[1]) ** 2); //Euclidean distance
	console.log("distance is:" + distance);
	return distance;

}
