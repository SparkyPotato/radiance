import graph.types;
import common;

[shader("pixel")]
public func main([vk::location(0)] data: u32) -> u32 {
    return data;
}
