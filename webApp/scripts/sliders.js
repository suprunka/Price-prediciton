var rangeSliderGrade= document.getElementById("rs-range-line-grade");
var rangeBulletGrade= document.getElementById("rs-bullet-grade");
var rangeSliderCondition = document.getElementById("rs-range-line-condition");
var rangeBulletCondition = document.getElementById("rs-bullet-condition");
var rangeSliderFloor = document.getElementById("rs-range-line-floor");
var rangeBulletFloor = document.getElementById("rs-bullet-floor");

rangeSliderCondition.addEventListener("input", showSliderValueCondition, false);

rangeSliderGrade.addEventListener("input", showSliderValueGrade, false);
rangeSliderFloor.addEventListener("input", showSliderValueFloor, false);

function showSliderValueGrade() {
  rangeBulletGrade.innerHTML = rangeSliderGrade.value;
  var bulletPosition = (rangeSliderGrade.value /rangeSliderGrade.max);
  rangeBulletGrade.style.left = (bulletPosition*136 ) + "px";
}
function showSliderValueCondition() {
  rangeBulletCondition.innerHTML = rangeSliderCondition.value;
  var bulletPosition = (rangeSliderCondition.value /rangeSliderCondition.max);
  rangeBulletCondition.style.left = (bulletPosition*136 ) + "px";
}
function showSliderValueFloor() {
  rangeBulletFloor.innerHTML = rangeSliderFloor.value;
  var bulletPosition = (rangeSliderFloor.value /rangeSliderFloor.max);
  rangeBulletFloor.style.left = (bulletPosition*136 ) + "px";
}