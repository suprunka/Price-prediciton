
  var zipcodes = ["98101", "98102", "98103", "98104", "98105", "98106", "98107", "98108", "98109", "98110", "98112", "98115", "98116", "98117", "98118", "98119", "98121", "98122", "98125", "98126", "98131", "98133", "98134", "98136", "98144", "98146", "98148", "98154", "98155", "98158", "98161", "98164", "98166", "98168", "98174", "98177", "98178", "98188", "98198", "98199", "98111", "98113", "98114", "98124", "98127", "98138", "98139", "98141", "98145", "98160", "98165", "98175", "98194", "98129", "98170", "98181", "98185", "98190", "98191", "98195"];

function addInput(divName) {
    var select = $("<select/>")
    $.each(zipcodes, function(a, b) {
        select.append($("<option/>").attr("value", b).text(b));
    });
    $("#" + divName).append(select);
}

addInput("zipcode")


function confirmAddress() {
geocode(platform)
}
function set_coordinates(x, y ) {
  $("#long").val(x);
  $('#lat').val(y)
}



function geocode(platform) {
    var address =  $("#address").val()
  var geocoder = platform.getGeocodingService(),
    geocodingParameters = {
      searchText: address,
      jsonattributes : 1
    };

  geocoder.geocode(
    geocodingParameters,
    onSuccess,
    onError
  );
}
function onSuccess(result) {
  var locations = result.response.view[0].result;

  addLocationsToMap(locations);
  // ... etc.
}

/**
 * This function will be called if a communication error occurs during the JSON-P request
 * @param  {Object} error  The error message received.
 */
function onError(error) {
  alert('Ooops!');
}

function addDraggableMarker(map, behavior, lat, lng){

  var marker = new H.map.Marker({lat:lat, lng:lng});
  // Ensure that the marker can receive drag events
  marker.draggable = true;
  map.addObject(marker);

  // disable the default draggability of the underlying map
  // when starting to drag a marker object:
  map.addEventListener('dragstart', function(ev) {
    var target = ev.target;
    if (target instanceof H.map.Marker) {
      behavior.disable();
    }
  }, false);


  // re-enable the default draggability of the underlying map
  // when dragging has completed
  map.addEventListener('dragend', function(ev) {
    var target = ev.target;
    if (target instanceof mapsjs.map.Marker) {
      behavior.enable();
	  var coord = map.screenToGeo(ev.currentPointer.viewportX,
            ev.currentPointer.viewportY);
        set_coordinates(coord.lng, coord.lat)

    }


  }, false);

  // Listen to the drag event and move the position of the marker
  // as necessary
   map.addEventListener('drag', function(ev) {
    var target = ev.target,
        pointer = ev.currentPointer;
    if (target instanceof mapsjs.map.Marker) {
      target.setPosition(map.screenToGeo(pointer.viewportX, pointer.viewportY));
    }
  }, false);
}

/**
 * Boilerplate map initialization code starts below:
 */

//Step 1: initialize communication with the platform
var platform = new H.service.Platform({
    'app_id': 'PWafOeMEykYIIGcoYB5V',
  'app_code': 'XRoNL4BRhvz80DDJynyhwg',
});
var pixelRatio = window.devicePixelRatio || 1;
var defaultLayers = platform.createDefaultLayers({
  tileSize: pixelRatio === 1 ? 256 : 512,
  ppi: pixelRatio === 1 ? undefined : 320
});

//Step 2: initialize a map - this map is centered over Boston
var map = new H.Map(document.getElementById('map'),
  defaultLayers.normal.map,{
  center: {lat:47.5, lng:-122.2},
  zoom: 12,
  pixelRatio: pixelRatio
});

//Step 3: make the map interactive
// MapEvents enables the event system
// Behavior implements default interactions for pan/zoom (also on mobile touch environments)
var behavior = new H.mapevents.Behavior(new H.mapevents.MapEvents(map));

// Step 4: Create the default UI:
var ui = H.ui.UI.createDefault(map, defaultLayers, 'en-US');

// Add the click event listener.
function addLocationsToMap(locations){
  var group = new  H.map.Group(),
    i;
    var x= locations.length-1
  // Add a marker for each location found

      set_coordinates(locations[x].location.displayPosition.longitude, locations[x].location.displayPosition.latitude);

      var marker3 = addDraggableMarker(map, behavior,locations[x].location.displayPosition.latitude, locations[x].location.displayPosition.longitude);
    map.setCenter({lat:locations[x].location.displayPosition.latitude, lng:locations[x].location.displayPosition.longitude});
    group.addObject(marker3);


  group.addEventListener('tap', function (evt) {
    map.setCenter(evt.target.getPosition());
    openBubble(
       evt.target.getPosition(), evt.target.label);
  }, false);

  // Add the locations group to the map
  map.addObject(group);
  map.setCenter(group.getBounds().getCenter());
}


function addLocationsToAddress(locations){
 $("address").val(locations);

}

// Now use the map as required...
addDraggableMarker(map, behavior, 47.5, -122.2);
geocode(platform);

