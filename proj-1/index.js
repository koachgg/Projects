var randomNumber1 = Math.floor(Math.random()*6)+1;

// math.random generates a random number from 0 to 0.999
// math.floor round off that number

// console.log(randomNumber1);

var randomDice = "dice" + randomNumber1 + ".png"; // dice1.png - dice6.png

var randomImg = "images/" + randomDice // images/dice1.png - images/dice6.png

var image1 = document.querySelectorAll("img")[0];

image1.setAttribute("src",randomImg);

// document.querySelector(".img1").setAttribute("src","images/dice" + randomNumber1 + ".png");

var randomNumber2 = Math.floor(Math.random()*6)+1;

var randomDice2 = "dice" + randomNumber2 + ".png"; // dice1.png - dice6.png

var randomImg2 = "images/" + randomDice2 // images/dice1.png - images/dice6.png

var image2 = document.querySelectorAll("img")[1];

image2.setAttribute("src",randomImg2);
// document.querySelector(".img2").setAttribute("src","images/dice"+ randomNumber2 + ".png");

// alert("working ig")
if (randomNumber1>randomNumber2)
{
  document.querySelector("h1").innerHTML="Player 1 wins";
}

else if (randomNumber1==randomNumber2) {
    document.querySelector("h1").innerHTML = "DRAW"
}

else{
  document.querySelector("h1").innerHTML = "Player 2 wins";
}
