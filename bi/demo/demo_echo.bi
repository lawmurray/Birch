/**
 * Echo a message to terminal.
 * 
 * - `message` : The message.
 */
program demo_echo(message:String <- "") {
  stdout.printf("%s\n", message);
}
