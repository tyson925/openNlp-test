package uy.com.collokia.opennlp

import opennlp.tools.namefind.NameFinderME
import opennlp.tools.namefind.NameSampleDataStream
import opennlp.tools.namefind.TokenNameFinderFactory
import opennlp.tools.namefind.TokenNameFinderModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.PlainTextByLineStream
import opennlp.tools.util.Span
import opennlp.tools.util.TrainingParameters
import java.io.BufferedOutputStream
import java.io.FileOutputStream
import java.nio.charset.Charset
import java.util.*




class OpenNLPTest {

    companion object {

        val sentences = arrayOf(
                "If President John F. Kennedy, after visiting France in 1961 with his immensely popular wife, famously described himself as 'the man who had accompanied Jacqueline Kennedy to Paris,' Mr. Hollande has been most conspicuous on this state visit for traveling alone.",
                "Mr. Draghi spoke on the first day of an economic policy conference here organized by the E.C.B. as a sort of counterpart to the annual symposium held in Jackson Hole, Wyo., by the Federal Reserve Bank of Kansas City. ",
                "Donald Trump's defenders have always argued that the entire notion his campaign colluded with Russia was all smoke and no fire.")

        //// http://opennlp.sourceforge.net/models-1.5/en-ner-person.bin
        val personModelFile = "en-ner-person.bin"
        val locationModelFile = "en-ner-location.bin"
        val corpusTrainFile = "corpus.train"
        val customModelFile = "en-ner-custom.bin"

        @JvmStatic
        fun main(args: Array<String>) {
        //    createCustomModel()
            userNameFinder()
        }


        fun createCustomModel() {
            println("Creating model...")
            val classLoader = Thread.currentThread().contextClassLoader
            val charset = Charset.forName("UTF-8")
            val lineStream = PlainTextByLineStream(
                    { classLoader.getResourceAsStream(corpusTrainFile) }, charset)
            val sampleStream = NameSampleDataStream(lineStream)

            val model = try {
                NameFinderME.train("en", "custom", sampleStream, TrainingParameters.defaultParams(), TokenNameFinderFactory())

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

    }
}