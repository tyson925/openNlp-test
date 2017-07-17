package uy.com.collokia.opennlp

import opennlp.tools.formats.ontonotes.OntoNotesNameSampleStream
import opennlp.tools.namefind.NameFinderME
import opennlp.tools.namefind.TokenNameFinderEvaluator
import opennlp.tools.namefind.TokenNameFinderFactory
import opennlp.tools.namefind.TokenNameFinderModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.ObjectStreamUtils
import opennlp.tools.util.Span
import opennlp.tools.util.TrainingParameters
import weka.core.tokenizers.NGramTokenizer
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.Paths
import java.util.*






class OpenNLPTest {

    companion object {

        val sentences = arrayOf(
                "If President John F. Kennedy, after visiting France in 1961 with his immensely popular wife, famously described himself as 'the man who had accompanied Jacqueline Kennedy to Paris,' Mr. Hollande has been most conspicuous on this state visit for traveling alone.",
                "Mr. Draghi spoke on the first day of an economic policy conference here organized by the E.C.B. as a sort of counterpart to the annual symposium held in Jackson Hole, Wyo., by the Federal Reserve Bank of Kansas City. ",
                "Donald Trump's defenders have always argued that the entire notion his campaign colluded with Russia was all smoke and no fire.")

        // http://opennlp.sourceforge.net/models-1.5/en-ner-person.bin
        val personModelFile = "models/en-ner-person.bin"
        // http://opennlp.sourceforge.net/models-1.5/en-ner-location.bin
        val locationModelFile = "models/en-ner-location.bin"
        val customModelFile = "models/en-ner-custom-onac.bin"

        val validatedTrainingFolders = listOf("training/spoken", "training/written_1", "training/written_2")

        @JvmStatic
        fun main(args: Array<String>) {
//            createCustomModel()
//            userNameFinder()
            evaluateCustomModel()
            //userCustomFinder()
        }

        fun getTrainingFiles(): List<String> {
            val classLoader = Thread.currentThread().contextClassLoader
            val files = mutableListOf<String>()
            validatedTrainingFolders.map {
                Paths.get(classLoader.getResource(it).path)
            }.forEach {
                Files.walk(it).filter { Files.isRegularFile(it) }.forEach {
                    files.add(it.toString())
                }
            }
            return files
        }


        fun createCustomModel() {
            println("Creating model...")
            val trainingFiles = getTrainingFiles().map {
                val file = File(it)
                file.useLines {
                    it.toList()
                }
            }.flatten()
            val lineStream = ObjectStreamUtils.createObjectStream(trainingFiles)
            val sampleStream = OntoNotesNameSampleStream(lineStream)

            val model = try {
                NameFinderME.train("en", null, sampleStream,
                        TrainingParameters.defaultParams(), TokenNameFinderFactory())

            } finally {
                sampleStream.close()
            }
            var modelOut: BufferedOutputStream? = null
            try {

                modelOut = BufferedOutputStream(FileOutputStream(customModelFile))
                model!!.serialize(modelOut)
                println("Model Created!")
            } finally {
                if (modelOut != null)
                    modelOut.close()
            }

        }

        fun userNameFinder() {
            val classLoader = Thread.currentThread().contextClassLoader
            classLoader.getResourceAsStream(personModelFile).use { personModelIn ->
                val personModel = TokenNameFinderModel(personModelIn)
                val personFinder = NameFinderME(personModel)
                classLoader.getResourceAsStream(locationModelFile).use { locationModelIn ->
                    val locationModel = TokenNameFinderModel(locationModelIn)
                    val locationFinder = NameFinderME(locationModel)
                    classLoader.getResourceAsStream(customModelFile).use { customModelIn ->
                        val customModel = TokenNameFinderModel(customModelIn)
                        val customFinder = NameFinderME(customModel)

                        val tokenizer = SimpleTokenizer.INSTANCE

                        sentences.forEach {
                            val tokens = tokenizer.tokenize(it)
                            val peopleSpans = personFinder.find(tokens)
                            val locationsSpans = locationFinder.find(tokens)
                            val customSpans = customFinder.find(tokens)
                            println("People")
                            println(Arrays.toString(Span.spansToStrings(peopleSpans, tokens)))
                            println("Locations")
                            println(Arrays.toString(Span.spansToStrings(locationsSpans, tokens)))
                            println("Customs")
                            println(Arrays.toString(Span.spansToStrings(customSpans, tokens)))
                        }
                    }
                }
                println("Done")
            }
        }

        fun evaluateCustomModel() {
            val classLoader = Thread.currentThread().contextClassLoader
            classLoader.getResourceAsStream(customModelFile).use { customModelIn ->
                val customModel = TokenNameFinderModel(customModelIn)
                val evaluator = TokenNameFinderEvaluator(NameFinderME(customModel))

                val trainingFiles = getTrainingFiles().map {
                    val file = File(it)
                    file.useLines {
                        it.toList()
                    }
                }.flatten()
                val testData = trainingFiles.toMutableList().shuffle().take(trainingFiles.size/4)

                val lineStream = ObjectStreamUtils.createObjectStream(testData)
                val sampleStream = OntoNotesNameSampleStream(lineStream)
                evaluator.evaluate(sampleStream)

                println(evaluator.fMeasure.toString())


            }
        }

        fun userCustomFinder() {
            val classLoader = Thread.currentThread().contextClassLoader
            classLoader.getResourceAsStream(customModelFile).use { customModelIn ->
                val customModel = TokenNameFinderModel(customModelIn)
                val customFinder = NameFinderME(customModel)

                val tokenizer = SimpleTokenizer.INSTANCE

                val sentence = """PARIS — They were handshake rivals before President Trump said the United States would withdraw from the Paris climate accord, and his relationship with President Emmanuel Macron of France didn’t seem to get any better after that awkward beginning.

But Mr. Trump and Mr. Macron appear to have put their strange and tense initial relationship behind them, in the service of a working partnership and the love of a parade.

Mr. Trump arrived in Paris just after 8:30 a.m. on Thursday, beginning his second European trip in two weeks. The visit was set in motion by a call Mr. Macron had made to discuss Syria, in which he invited Mr. Trump to Bastille Day celebrations on July 14. The president and the first lady, Melania Trump, landed at Paris Orly Airport on Air Force One to the reception of a 10-car motorcade.

Mr. Trump loves the trappings of the presidency, whether in the United States or in another country. That includes occupying the most prestigious seats at the Bastille Day ceremony, a pomp-filled parade steeped in military tradition and hardware.

Continue reading the main story
RELATED COVERAGE


Emmanuel Macron to Welcome Trump, an Unlikely Partner, to France JULY 12, 2017

In Lofty Versailles Speech, Macron Tells the French to Prepare for Change JULY 3, 2017

Macron Quickly Assumes a Presidential Attitude MAY 30, 2017

Opinion Op-Ed Contributors
The Trump Vision for America Abroad JULY 13, 2017

Opinion Contributing Op-Ed Writer
When Trump Meets Jupiter in Paris JULY 12, 2017
RECENT COMMENTS

njglea 36 minutes ago
The Con Don and Mr. Macron both seem to want a corporate world without any "government" interference. WE THE PEOPLE apparently were meant...
pieceofcake 1 hour ago
'In France, Trump and Macron Strive to Put Awkward Start Behind Them'As Macron just wanted to prove that Jim -(aka 'Trump') still goes to...
Phyliss Kirk 1 hour ago
Stop using the word "meddling" for Russia's cyber attack on our election. This is cyberwar to put out propaganda and use psychological...
SEE ALL COMMENTS  WRITE A COMMENT
Mr. Trump, for his own inaugural parade, had expressed a desire to include tanks and fighter jets. That wish was not granted, but Mr. Trump remains transfixed by displays of military power.

He arrives in Europe once again leaving behind a trail of questions related to Russian meddling in the 2016 election, flying to the more welcoming arms of a foreign leader with whom his bond is still fragile.

Mr. Macron and Mr. Trump have had an unusual relationship, characterized in public primarily by a few forceful, awkward handshakes, particularly their first, which Mr. Macron made clear was an effort to show the American president that he could not be bullied.

So Mr. Trump’s decision to accept the invitation startled some of his aides.

For the embattled American president, trips overseas — the visit to France will be his third abroad in two months — have been a surprising pleasure, a reprieve from days filled with cable news coverage of the Russia investigation and swirling questions of whether his campaign aides worked in concert with the foreign power.

For Mr. Macron, who took office in May, the visit is a chance to establish himself, if only by default, as Mr. Trump’s first point of contact in Western Europe, at a time when Britain is distracted by its plans to leave the European Union and Germany is focused on national elections in the fall.

It is an unlikely partnership, given Mr. Trump’s stated admiration for Marine Le Pen, the far-right populist whom Mr. Macron defeated in May, and the leaders’ radically different world views. Mr. Macron is a pro-European technocrat who admires Silicon Valley, and Mr. Trump an America-first nationalist who is skeptical of multilateral institutions like the European Union.

Mr. Trump’s visit to Paris began with an airport arrival ceremony. He then attended a meeting with troops at the American ambassador’s residence while Mrs. Trump toured the Necker children’s hospital.

“I always say how important it is to have, you know, teachers in children’s lives. It’s the most important,” Mrs. Trump said. “They see them every day and spend so much time. It’s very important in the child’s life.”

“You look very good. Very strong,” Mrs. Trump told a 14-year-old girl in a wheelchair. “One day you will be walking and running.”

At the United States ambassador’s residence, Mr. Trump joined a lunch that was also attended by Mike Pompeo, the C.I.A. director; Lt. Gen. H. R. McMaster, the national security adviser; and Gen. Joseph F. Dunford Jr., the chairman of the Joint Chiefs of Staff. The president also addressed military personnel and their families, before departing for the Hôtel National des Invalides, a sprawling patchwork of museums that includes the tomb of Napoleon Bonaparte.

Mr. and Mrs. Trump were feted with a welcome ceremony that included more than two-dozen horses carrying men in uniforms. Mr. Trump and Mr. Macron strolled through the expansive courtyard, a soldier carrying a sword behind them. At one point, Mr. Macron put his right hand on Mr. Trump’s back as he used his left hand to point to a columned facade.

Later, they were scheduled to have a closed-door meeting at the Élysée, the presidential palace, followed by a joint news conference. The men will cap the day with a dinner at Le Jules Verne, the elite, blue-lobster-serving restaurant ensconced in the Eiffel Tower.

That meal is something of a surprise, considering Mr. Trump’s fondness for ketchup-doused steak and cheeseburgers rather than gourmet foods.

Mr. Trump has been assured a premium spot at the parade on Friday, before he returns to the United States midday."""


                val tokens = tokenizer.tokenize(sentence)
                val customSpans = customFinder.find(tokens)
                val types = customSpans.map { it.type }
                types.zip(Span.spansToStrings(customSpans, tokens)).forEach {
                    println("${it.first}: ${it.second}")
                }


                println("""ngrams

                """)
                val ngramTokenizer = NGramTokenizer()
                ngramTokenizer.nGramMinSize = 1
                ngramTokenizer.nGramMaxSize = 3
                val ngramTokens = NGramTokenizer.tokenize(ngramTokenizer, arrayOf(sentence))
                val ngramCustomSpans = customFinder.find(ngramTokens)
                val ngramTypes = ngramCustomSpans.map { it.type }
                ngramTypes.zip(Span.spansToStrings(ngramCustomSpans, ngramTokens)).forEach {
                    println("${it.first}: ${it.second}")
                }

            }
            println("Done")
        }

    }
}

fun <T:Comparable<T>> MutableList<T>.shuffle():List<T>{
    val rg : Random = Random()
    for (i in 0..size - 1) {
        val randomPosition = rg.nextInt(size)
        val tmp : T = this[i]
        this[i] = this[randomPosition]
        this[randomPosition] = tmp
    }
    return this
}