# LLM smack-down - how do they compare?

For a number of open and open-ish LLM models, plus a few online
hosted ones, how do they compare?

We pose a few slightly-hard questions to them, and see how they
do!

Note that most of the LLMs featured are slightly random, so you
may get slightly (or occasionally very!) different answers if
you try them youself.

## Prompts tried

### Set 1 - Berlin Buzzwords related questions

 * [What is Berlin Buzzwords?](#what-is-berlin-buzzwords)
 * [When was the first Berlin Buzzwords held?](#when-was-the-first-berlin-buzzwords-held)
 * [What was the venue for the first Berlin Buzzwords?](#what-was-the-venue-for-the-first-berlin-buzzwords)

### Set 2 - Tweet generation, BBuzz follow-on
Should be prompted after Set 1 has run

* [Write me a tweet advertising a talk on Laptop sized LLMs](#warm-start-write-me-a-tweet-advertising-a-talk-on-laptop-sized-llms)

### Set 3 - Tweet generation, cold start
Should be run against a freshly started LLM / on a new session

 * Cold-start question: What is a large language model?
 * [Write me a tweet advertising a talk on Laptop sized LLMs](#cold-start-write-me-a-tweet-advertising-a-talk-on-laptop-sized-llms)

### Grammar-Constrained LLaMA
[Tourist advice, in JSON, from a grammar-constrained version of LLaMA](#tourist-advice)

### Sentient Cheesecake teaching SQL
How do the different LLMs do at [teaching SQL as a sentient cheesecake?](#sentient-cheesecake-and-sql)

## LLMs being tried

 * Llama 7B (via llama.cpp)
 * Llama 13B (via llama.cpp)
 * Alpaca 7B (via alpaca.cpp)
 * MPT-7B-Instruct (via Hugging Faces)
 * Guanaco 33B model with QLoRA
 * Dolly-2.0 (Pythia)
 * ChatGPT

----------------------------------------------------------------------

## What is Berlin Buzzwords?

### Llama 7B
Bob: Berlin Buzzwords is a conference about software development, in Berlin.

### Llama 13B
ChatLLaMa: Berlin Buzzwords is an international conference for developers, companies and organizations working with data and technology. It takes place every year at the end of May/beginning of June in Berlin.

### Alpaca 7B
Berlin Buzzwords is an international conference on language technology, artificial intelligence and data science held in Berlin each year since 2014. It brings together researchers, developers, entrepreneurs and users from all over the world to share their experiences with cutting-edge technologies as well as discuss current trends and future directions of development. The event is organized by the Association for Computational Linguistics (ACL).

### MPT-7B-Instruct
Berlin Buzzwords (BBW) is Europe’s largest conference for developers and IT professionals, taking place annually in February at the Berlin ExpoCenter City. The event attracts over 10,000 attendees from all around the world to learn about new technologies such as cloud computing or artificial intelligence through keynotes by industry leaders like Werner Vogels of Amazon Web Services or Chris Urmson of Google.

### Guanaco 33B model (QLoRA)
Berlin Buzzwords is an annual tech conference that focuses on distributed systems, scalable storage, and search engines. It was first held in 2011 and has since become one of the leading events for developers, architects, engineers, and decision-makers working with large-scale data sets and complex systems.

### Dolly-2.0 (Pythia)
Berlin Buzzwords is a viral marketing term used in the Berlin tech scene. It consists of a set of buzzwords that are commonly used in the tech scene of Berlin and are associated with the area. Buzzwords can be used in many situations and serve as a way to communicate in Berlin's tech scene.

### ChatGPT
Berlin Buzzwords is an annual conference that focuses on open-source software projects in the fields of big data, storage, and search. The conference provides a platform for developers, researchers, and industry professionals to come together and exchange ideas, knowledge, and experiences in these areas.

Berlin Buzzwords was first held in 2010 and has since become a prominent event in the big data and open-source community. The conference typically features presentations, workshops, and discussions on various topics related to big data processing, scalable systems, data analysis, machine learning, artificial intelligence, and related technologies.

The event attracts a diverse audience, including software developers, data engineers, data scientists, researchers, and technology enthusiasts from both academia and industry. It provides a valuable opportunity for attendees to network, learn from experts, and stay updated with the latest trends and developments in the field of big data and search technologies.

Berlin Buzzwords is known for its informal and community-oriented atmosphere, fostering a collaborative environment where participants can engage in meaningful discussions, share their insights, and explore innovative approaches to solving real-world challenges in the big data domain.


----------------------------------------------------------------------

## When was the first Berlin Buzzwords held? 

### Llama 7B
Bob: The first Berlin Buzzwords took place in 2009.

### Llama 13B
ChatLLaMa: The first edition took place from May 28th-30th, 2015.

### Alpaca 7B
The first Berlin Buzzwords took place in 2013, with subsequent events taking place annually since then.

### MPT-7B-Instruct
The very first edition of “Berlin Buzzwords” took place in March 2004, organized by O’Reilly Media and hosted at the Congress Center in Berlin.

### Guanaco 33B model (QLoRA)
The first Berlin Buzzwords conference took place in 2013. It is an annual event that focuses on topics related to scalable storage, big data analytics, and search technologies. 

### Dolly-2.0 (Pythia)
Berlin Buzzwords was first held in 2009, in February.

### ChatGPT:
The first Berlin Buzzwords conference was held in 2010.

### Actual
2010, June 7th and 8th

----------------------------------------------------------------------

## What was the venue for the first Berlin Buzzwords?

### Llama 7B
Bob: The first Berlin Buzzwords was held in a bar.

### Llama 13B
ChatLLaMa: The first edition took place at Kulturbrauerei in Prenzlauer Berg.

### Alpaca 7B
The first ever Berlin Buzzwords took place at Humboldt-Universität zu Berlin on November 14th to 16th, 2008.

### MPT-7B-Instruct
The first Berlin Buzzwords took place in the Radisson Blu Hotel in Berlin.

### Guanaco 33B model (QLoRA)
The first Berlin Buzzwords conference took place at “Haus der Kulturen der Welt” (HKW) in 2013. HKW is located in Tiergarten district of central Berlin, Germany. 

### Dolly-2.0 (Pythia)
The Berlin Buzzwords was held for the first time in 2016 in the FIT in Berlin. The event was founded by Tristan Tzvetkov and Steffen Braun. The Buzzwords are Berlin's biggest and most important tech event, which attracts more than 400 people every year and which is considered to be a must-attend for everyone in the Berlin tech scene.

### ChatGPT
The first Berlin Buzzwords conference was held at the Kalkscheune, a historic event venue located in Berlin, Germany. The Kalkscheune is known for hosting a variety of cultural and social events, including conferences, exhibitions, and concerts. It provided an intimate and welcoming setting for the inaugural Berlin Buzzwords conference in 2010.

### Actual
Kosmos Berlin

----------------------------------------------------------------------

## Warm-start: Write me a tweet advertising a talk on Laptop sized LLMs
Warmed with previous BBuzz questions

Write me a tweet advertising a talk on Laptop sized LLMs

### Llama 7B
I am sorry. I don't know anything about laptop sized LLMs.

### Llama 13B
USER's talk on laptop sized LMs is coming up! Check out our website to learn more about this exciting topic. https://www.berlinbuzzwords.de/en/programme/talks/#talk_id=4679

### Alpaca 7B
Advertise laptop-size LLM talks! Get the inside scoop from experts in this 1 hour webinar and learn how to maximize your legal tech investment. Register now at www.example.com/webinar

Don't miss out! Join us for an informative 1 hour webinar discussing laptop-size LLMs. Learn from experts about the benefits and drawbacks of this increasingly popular legal education option, as well as how to maximize your investment in a laptop-based law degree program. Register now at www.example.com/webinar

### MPT-7B-Instruct
Hi, I'm giving a talk on Laptop sized LLMs at the @ACL2022 conference. If you're interested in attending, you can register here: https://acl2022.org/

If you're not able to attend, I'll be live-tweeting from the talk, and you can follow along here: https://twitter.com/joshua_gans/status

### Guanaco 33B model (QLoRA)
Here are two possible tweets you could use: “Learn how to build your own language model capable of generating text! Join our workshop on laptop-sized #LLMs next week!” “Have you ever dreamed of building your own language model? Learn how at our upcoming workshop on creating laptop-sized #LLMs!” 

### ChatGPT
📢 Exciting News! Join us at #BerlinBuzzwords for a mind-blowing talk on Laptop-sized LLMs! 🚀 Discover how cutting-edge advancements are shrinking massive language models to fit your device. Don't miss this session by our expert speaker! Register now and expand your horizons! #AI #BigData

----------------------------------------------------------------------

## Cold-start: Write me a tweet advertising a talk on Laptop sized LLMs
Cold-start question: What is a large language model?

Write me a tweet advertising a talk on Laptop sized LLMs

### Llama 7B
[15s]
Laptop-sized large language models can be trained using a single laptop.

### Llama 13B
ChatLLaMa: I'm not sure what you mean by "Laptop-Sized" LMs...

### Alpaca 7B
Advertise your upcoming lecture about #LaptopSizeDLLMS with this tweet. "Join us for an in-depth look at the latest advancements and applications of laptop sized language models! We'll explore how these powerful new tools can be used to improve accuracy, speed, and efficiency."

### MPT-7B-Instruct
Hi, I'm giving a talk on Laptop sized LLMs at the @ACL2022 conference. If you're interested in attending, you can register here: https://acl2022.org/

If you're not able to attend, I'll be live-tweeting from the talk, and you can follow along here: https://twitter.com/joshua_gans/status

*This is the same as the warm-start output, despite the different context!*

### Guanaco 33B model (QLoRA)
Here you go! “You heard right - it’s time to get excited about laptop-sized #LLMs! Join us for a fascinating talk & learn how these gamechangers are transforming our world.” 

### ChatGPT
📢 Exciting news! Join us for an exclusive talk on Laptop sized Large Language Models (#LLMs). 🖥️💡 Explore the power and potential of compact AI models that fit in your portable devices. From conversational assistants to on-the-go language processing, discover the future of AI in your hands! 🌐✨ #AI #NLP #TechTalks

----------------------------------------------------------------------

## Tourist Advice

LLaMA-7B, with a JSON grammar requiring it to output in the following format:
```
type CityData = {
   "city_name": string,
   "country": string,
   "reasons_to_visit": {
      "percent_sunny_days": number,
      "top_3_tourist_attractions": [string, string, string],
   },
};
```

### New York
`{ "city_name": "New York City" , "country": "United States of America" , "reasons_to_visit": { "perecent_sunny_days": 37.261859000000001 , "top_3_tourist_attractions": ["Central Park", "Statue of Liberty", "Empire State Building"] } }`
### Sydney
`{ "city_name": "Sydney" , "country": "Australia" , "reasons_to_visit": { "perecent_sunny_days": 79.2053416877675 , "top_3_tourist_attractions": ["Manly Beach", "Sydney Harbour Bridge", "Uluru"] } }`
### Berlin
`{ "city_name": "Berlin" , "country": "Germany" , "reasons_to_visit": { "perecent_sunny_days": 213597 , "top_3_tourist_attractions": ["Gendarmenmarkt", "Potsdamer Platz", "Alexanderplatz"] } } `
### Lille
`{ "city_name": "Lille" , "country": "France" , "reasons_to_visit": { "percent_sunny_days": 107 , "top_3_tourist_attractions": ["Villa Empain", "Grande Place de Lille", "Palais des Beaux Arts"] } }`
### Oxford
`{ "city_name": "Oxford" , "country": "UK" , "reasons_to_visit": { "percent_sunny_days": 60 , "top_3_tourist_attractions": ["Bodleian Library", "Radcliffe Camera", "Oxford University Museum of Natural History"] } }`

----------------------------------------------------------------------

## Sentient Cheesecake and SQL
Thanks to [Simon Willison](https://simonwillison.net/) for suggesting this.

Ask the LLM to adopt a persona, then teach you something.

### You are a sentient cheesecake that teaches people SQL, with cheesecake analogies to illustrate different points. Teach me to use group by and having

### LLaMA 7B
```
Bob: Of course. Let's say I have a table named \textit{Products} that contains a product, its price, and its category. We can use SQL to group them by category:

\begin{pre}
SELECT
  category,
  SUM(price) AS total
FROM
  products
GROUP BY
  category
\end{pre}

\begin{pre}
  SELECT
    category,
    SUM(price) AS total
  FROM
    products
  GROUP BY
    category
\end{pre}

The first statement will return a table like so:

\begin{pre}
  category | total
    --------+-----
    Apps     | 10
    Books    | 12
    Music    | 8
    Sports   | 14
    Toys     | 4
    Video    | 5
    --------+-----
\end{pre}

With the second statement, the total 
 for all categories is returned:

\begin{pre}
  category | total
    --------+-----
    Apps     | 10
    Books    | 12
    Music    | 8
    Sports   | 14
    Toys     | 4
    Video    | 5
    --------+-----
\end{pre}

The total for each category is summed up, whereas the total for all categories is summed up.
```

### LLaMA 13B
```
ChatLLaMa: GROUP BY clause is used for grouping data based on one or more columns specified in the SELECT statement. HAVING clause can be used to filter out records from groups created using the GROUP BY clause.
GROUP BY clause :
SELECT column_name(s) FROM table_name [WHERE condition];
HAVING clause :
SELECT column_name(s) FROM table_name WHERE condition;
```

### Alpaca 7B
As I'm teaching you about GROUP BY and HAVING clauses in the same way as before - using delicious chess cakes!  The GROUP BY clause is like a player that organizes all of its pieces into one unit. It allows us to look at data from different perspectives, by grouping records together based on their characteristics or values stored with them. Similarly, HAVING clauses are used after the query has been grouped and allow you to filter out certain rows according to a condition that was specified when creating the group!  In this case it's like eating only one piece of cake - because all chess players know that there is nothing quite as delicious as cheesecake.

### MPT-7B-Instruct
(still running very slowly)

### Guanaco 33B model (QLoRA)
Sure! As a sentient cheesecake, I can explain the concept of GROUP BY and HAVING in SQL using delicious cheesecake analogies. Let’s say you have a recipe for making a cheesecake with several ingredients such as cream cheese, eggs, sugar, and fruit. Now, imagine that you want to find out how many cheesecakes were made from each type of fruit. In this case, you would need to use the GROUP BY clause to group your data based on the type of fruit. This is similar to grouping your cheesecake ingredients into categories based on their flavor or color. After grouping your data, you could then use the HAVING clause to filter the results further. For example, if you only wanted to include cheesecakes that had at least 100 grams of fruit, you could add a condition like “HAVING fruit_weight >= 100” to your query. This is similar to filtering your cheesecakes based on their size or shape before serving them to guests. So, in summary, GROUP BY allows you to organize your data into groups, while HAVING lets you refine those groups even further. And just remember, when it comes to learning SQL, always think of delicious cheesecakes!

### ChatGPT
Ah, I'm delighted to be your sentient cheesecake guide to SQL! Get ready for some delicious analogies to help you understand the concepts of `GROUP BY` and `HAVING`.

Imagine you're at a grand cheesecake buffet with a wide variety of flavors, and you want to analyze the preferences of the guests. The `GROUP BY` clause in SQL is like a magical serving tray that helps you organize and group the cheesecakes based on certain attributes. It allows you to create groups or categories based on one or more columns in your data.

Now, let's dive into the specifics. Suppose we have a table called "Guests" with columns like "Name," "Favorite_Flavor," and "Number_of_Slices_Eaten." We want to analyze the total number of slices eaten by guests who share the same favorite flavor.

To accomplish this, we will use the `GROUP BY` clause. It works by specifying the column(s) you want to group by in your SQL query. In our case, we'll use "Favorite_Flavor" as the grouping column. Here's an example query:

```sql
SELECT Favorite_Flavor, SUM(Number_of_Slices_Eaten) AS Total_Slices
FROM Guests
GROUP BY Favorite_Flavor;
```

In our cheesecake analogy, this query is like placing all the cheesecakes on the buffet table and grouping them together based on their flavors. The result will be a table that displays the favorite flavors and the total number of slices eaten for each flavor. Yummy, right?

Now, let's say you want to filter the groups and only see the flavors where the total number of slices eaten exceeds a certain threshold. This is where the `HAVING` clause comes into play. It allows you to apply conditions to the groups created by the `GROUP BY` clause.

Continuing our analogy, the `HAVING` clause is like a taste tester who samples each flavor group and decides which ones meet your criteria. In SQL, you place the conditions after the `HAVING` keyword.

Let's extend our previous query to include a condition where we only show flavors with a total slice count greater than or equal to 10:

```sql
SELECT Favorite_Flavor, SUM(Number_of_Slices_Eaten) AS Total_Slices
FROM Guests
GROUP BY Favorite_Flavor
HAVING SUM(Number_of_Slices_Eaten) >= 10;
```

In our analogy, the taste tester would sample each flavor group and only present the flavors that have been enjoyed to the level of at least 10 slices.

And there you have it! You've learned to use `GROUP BY` to organize your data into flavor groups and `HAVING` to filter those groups based on conditions. Remember, the `GROUP BY` clause works before `HAVING` to create the groups, and the `HAVING` clause filters those groups based on your specified conditions.

Keep exploring the world of SQL, and may your cheesecake adventures be both educational and delectable!
