import java.io.File;
import java.lang.*;
import org.gearman.worker.*;
import org.gearman.util.*;
import org.gearman.common.*;
import org.gearman.client.*;

public class Biome extends AbstractGearmanFunction {

    public static cu MCSave;
    public static pb BioGen;

    public String getName() {
        System.out.println("getname");
        return "GetBiome";
    }

    public GearmanJobResult executeFunction() {
        //System.out.println("executing");
        String data = new String((byte[])this.data);
        //System.out.println("got data -->" + data + "<--");
        String[] s = data.split(",");

        int x = Integer.parseInt(s[0]);
        int y = Integer.parseInt(s[1]);

        BioGen.a(x,y,1,1);
        double temp = BioGen.a[0];
        double moisture = BioGen.b[0];

        String result = Double.toString(temp) + "/" + Double.toString(moisture);
        //System.out.println(result);
        GearmanJobResult gjr = new GearmanJobResultImpl(this.jobHandle,true, result.getBytes(),
                new byte[0], new byte[0], 0, 0);
        return gjr;
    }


    public static void main(String[] args) {

        System.out.println("Locating Minecraft save...");

        MCSave = new cu(new File("/home/achin/devel/overviewer-fork"), "world.test");

       /* if (MinecraftSave.q)
            System.out.println("Loading level...");
        else
        {   
            System.out.println("Failed to load level! Aborting.");
            return;
        }*/


        BioGen = new pb(MCSave);

       /* BiomeGenerator.a(0,1,1,1);
        double temp = BiomeGenerator.a[0];
        double moisture = BiomeGenerator.b[0];

        System.out.println("Got biome vals at (0,0)");
        System.out.println("Temperature: " + Double.toString(temp));
        System.out.println("Moisture: " + Double.toString(moisture));
        */


        org.gearman.worker.GearmanWorker w = new org.gearman.worker.GearmanWorkerImpl();

        w.addServer(new GearmanNIOJobServerConnection("localhost"));

        w.registerFunction(Biome.class);

        System.out.println("working...");
        w.work();

    }


}
