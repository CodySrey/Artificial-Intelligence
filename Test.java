import java.util.*;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.HashMap;
import java.util.HashSet;

public class Test{
    private static class States{
        String city;
        Map<States, Integer> neighbors = new HashMap<>();
        States(String city){
            this.city = city;
        }
    }
    private Map<String, States> cities = new HashMap<>();

    public void mapCity(){ //Creates cities and connects the nearby cities together
        addCity("Eforie");
        addCity("Hirsova");
        addCity("Urziceni");
        addCity("Bucharest");
        addCity("Giurgiu");
        addCity("Vaslui");
        addCity("Iasi");
        addCity("Neamt");
        connectCity("Eforie","Hirsova",86);
        connectCity("Hirsova","Urziceni",98);
        connectCity("Giurgiu","Bucharest",90);
        connectCity("Bucharest","Urziceni",85);
        connectCity("Urziceni","Vaslui",142);
        connectCity("Vaslui","Iasi",92);
        connectCity("Iasi","Neamt",87);
    }
    private void addCity(String city){ //Add makes a new city to cities
    }

    private void connectCity(String city1, String city2, int distance){ //Connect makes the two cities connect within the distance
        cities.put(city, new States(city));
        States node1 = cities.get(city1);
        States node2 = cities.get(city2);
        
        node1.neighbors.put(node2,distance);
        node2.neighbors.put(node1,distance);
    }

    public boolean cConnected(String city1, String city2){ //Checks to find the path between two cities
        return pathFinder(city1, city2) !=null;
    }

    public String pathFinder(String startCity, String endCity){ //Finds the shortest path, Integer is the distance you enter to check
        return pathFinderDistance(startCity, endCity, Integer.MAX_VALUE);
    }

    public boolean cConnectedDistance(String city1, String city2, int maxDistance){ //Checks if there's a path between two cities within an input for max distance
        return pathFinderDistance(city1, city2, maxDistance) !=null;
    }
    //BFS that finds the shortest path between two cities
    private String pathFinderDistance(String startCity, String endCity, int maxDistance){
        States startNode = cities.get(startCity);
        States endNode = cities.get(endCity);

        if(startNode == null || endNode == null){
            return null; //The city was not found
        }

        Queue<States> queue = new LinkedList<>();
        Set<States> went = new HashSet<>();
        Map<States, States> parent = new HashMap<>(); //Rebuilds the path
        queue.add(startNode);
        went.add(startNode);

        while(!queue.isEmpty()){
            States current = queue.poll();
            if(current == endNode){
                return rebuildPath(parent,startNode,endNode);
            }

            for(Map.Entry<States, Integer> neighborEntry : current.neighbors.entrySet()){
                States neighbor = neighborEntry.getKey();
                int distance = neighborEntry.getValue();

                if(!went.contains(neighbor) && distance <= maxDistance){
                    went.add(neighbor);
                    parent.put(neighbor, current);
                    queue.add(neighbor);
                }
            }
        }

        return null; //It means no paths were found
    }

    private String rebuildPath(Map<States,States> parent, States startNode, States endNode){ //Reconstructs the path from parent which is used to store the parent node during the BFS
        StringBuilder path = new StringBuilder(endNode.city);
        States current = endNode;

        while(current != startNode){
            current = parent.get(current);
            path.insert(0, current.city + " > ");
        }   

        return path.toString();
    }

    public static void main(String[] args){ //Tests the distances and see if the cities are displayed
        Test map = new Test();
        map.mapCity();

        System.out.println("Is there a path from Eforie to Neamt?" + " " + map.cConnected("Eforie", "Neamt"));
        System.out.println("The Path from Eforie to Neamt?" + " " + map.pathFinder("Eforie", "Neamt"));
        System.out.println();
        System.out.println("Is there a path from Urziceni to Neamt?" + " " + map.cConnected("Urziceni", "Neamt"));
        System.out.println("The Path from Urziceni to Neamt?" + " " + map.pathFinder("Urziceni", "Neamt"));
        System.out.println();
        System.out.println("Is there a path from Giurgiu to Neamt?"+ " " + map.cConnected("Giurgiu","Neamt"));
        System.out.println("The Path from Giurgiu to Neamt?" + " " + map.pathFinder("Giurgiu", "Neamt"));
        System.out.println();
        System.out.println("Is there a path from Bucharest to Iasi with a max distance of 100?" + " " + map.cConnectedDistance("Bucharest", "Iasi", 100));
        System.out.println("Path from Bucharest to Iasi within 100 max distance: " + map.pathFinderDistance("Bucharest", "Iasi", 100));
        System.out.println();
        System.out.println("Is there a path from Hirsova to Urziceni with a max distance of 100?" + " " + map.cConnectedDistance("Hirsova", "Urziceni", 100));
        System.out.println("Path from Hirsova to Urziceni within 100 max distance: " + map.pathFinderDistance("Hirsova", "Urziceni", 100));
    }
}