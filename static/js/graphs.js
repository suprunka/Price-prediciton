queue()
    .defer(d3.json, "/statistics")
    .await(makeGraphs);

function makeGraphs(error, projectsJson) {
	
	//Clean projectsJson data
	var donorschooseProjects = projectsJson;
//	var dateFormat = d3.time.format("%Y-%m-%d");
//	donorschooseProjects.forEach(function(d) {
//		d["date_posted"] = dateFormat.parse(d["date_posted"][0,8]);
//		d["date_posted"].setDate(1);
//		d["total_donations"] = +d["total_donations"];
//	});
//    console.log(dateFromat)

	//Create a Crossfilter instance
	var ndx = crossfilter(donorschooseProjects);

	//Define Dimensions
	var priceDim = ndx.dimension(function(d) { return d["price"]; });
	var bedroomsDim = ndx.dimension(function(d) { return d["bedrooms"]; });
	var bathroomsDim = ndx.dimension(function(d) { return d["bathrooms"]; });
	var sqftlivingDim = ndx.dimension(function(d) { return d["sqft_living"]; });
	var sqftlotDim  = ndx.dimension(function(d) { return d["sqft_lot"]; });
    print(bedroomsDim)

	//Calculate metrics
	var numProjectsByPrice = priceDim.group();
	var numProjectsByResourceType = bedroomsDim.group();
	var numProjectsByPovertyLevel = bathroomsDim.group();
	var totalDonationsByState = sqftlivingDim.group().reduceSum(function(d) {
		return d["price"];
	});
	var all = ndx.groupAll();
	var totalDonations = ndx.groupAll().reduceSum(function(d) {return d["price"];});
	var max_state = totalDonationsByState.top(1)[0].value;

	//Define values (to be used in charts)
	var minDate = priceDim.bottom(1)[0]["price"];
	var maxDate = priceDim.top(1)[0]["price"];

    //Charts
//	var timeChart = dc.barChart("#time-chart");
	var resourceTypeChart = dc.rowChart("#resource-type-row-chart");
//	var povertyLevelChart = dc.rowChart("#poverty-level-row-chart");
//	var usChart = dc.geoChoroplethChart("#us-chart");
//	var numberProjectsND = dc.numberDisplay("#number-projects-nd");
//	var totalDonationsND = dc.numberDisplay("#total-donations-nd");

//	numberProjectsND
//		.formatNumber(d3.format("d"))
//		.valueAccessor(function(d){return d; })
//		.group(all);

//	totalDonationsND
//		.formatNumber(d3.format("d"))
//		.valueAccessor(function(d){return d;})
//		.group(totalDonations)
//		.formatNumber(d3.format(".3s"));

//	timeChart
//		.width(600)
//		.height(160)
//		.margins({top: 10, right: 50, bottom: 30, left: 50})
//		.dimension(dateDim)
//		.group(numProjectsByPtivr)
//		.transitionDuration(500)
//		.x(d3.time.scale().domain([minDate, maxDate]))
//		.elasticY(true)
//		.xAxisLabel("Year")
//		.yAxis().ticks(4);

	resourceTypeChart
        .width(300)
        .height(250)
        .dimension(bedroomsDim)
        .group(numProjectsByResourceType)
        .xAxis().ticks(4);
//
//	povertyLevelChart
//		.width(300)
//		.height(250)
//        .dimension(povertyLevelDim)
//        .group(numProjectsByPovertyLevel)
//        .xAxis().ticks(4);


//	usChart.width(1000)
//		.height(330)
//		.dimension(stateDim)
//		.group(totalDonationsByState)
//		.colors(["#E2F2FF", "#C4E4FF", "#9ED2FF", "#81C5FF", "#6BBAFF", "#51AEFF", "#36A2FF", "#1E96FF", "#0089FF", "#0061B5"])
//		.colorDomain([0, max_state])
//		.overlayGeoJson(statesJson["features"], "state", function (d) {
//			return d.properties.name;
//		})
//		.projection(d3.geo.albersUsa()
//    				.scale(600)
//    				.translate([340, 150]))
//		.title(function (p) {
//			return "State: " + p["key"]
//					+ "\n"
//					+ "Total Donations: " + Math.round(p["value"]) + " $";
//		})

    dc.renderAll();

};