package uy.com.collokia.opennlp

import opennlp.tools.namefind.NameFinderME
import opennlp.tools.namefind.TokenNameFinderModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.Span
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

        @JvmStatic
        fun main(args: Array<String>) {
            userNameFinder()
        }


        fun userNameFinder(){
            val classLoader = Thread.currentThread().contextClassLoader
            classLoader.getResourceAsStream(personModelFile).use { personModelIn ->
                    val personModel = TokenNameFinderModel(personModelIn)
                    val personFinder = NameFinderME(personModel)
                classLoader.getResourceAsStream(locationModelFile).use { locationModelIn ->
                    val locationModel = TokenNameFinderModel(locationModelIn)
                    val locationFinder = NameFinderME(locationModel)
                    val tokenizer = SimpleTokenizer.INSTANCE

                    sentences.forEach {
                        val tokens = tokenizer.tokenize(it)
                        val peopleSpans = personFinder.find(tokens)
                        val locationsSpans  = locationFinder.find(tokens)
                        println("People")
                        println(Arrays.toString(Span.spansToStrings(peopleSpans, tokens)))
                        println("Locations")
                        println(Arrays.toString(Span.spansToStrings(locationsSpans, tokens)))
                    }
                }
            }
            println("Done")
        }
    }

}